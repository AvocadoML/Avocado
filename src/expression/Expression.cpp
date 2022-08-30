/*
 * Expression.cpp
 *
 *  Created on: Jul 27, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/expression/Expression.hpp>
#include <Avocado/expression/node_reference.hpp>

#include <Avocado/expression/nodes/activations.hpp>
#include <Avocado/expression/nodes/arithmetic.hpp>
#include <Avocado/expression/nodes/bitwise.hpp>
#include <Avocado/expression/nodes/comparison.hpp>
#include <Avocado/expression/nodes/compound.hpp>
#include <Avocado/expression/nodes/technical.hpp>
#include <Avocado/expression/nodes/reduction.hpp>
#include <Avocado/expression/nodes/functions.hpp>

#include <Avocado/utils/instanceof.hpp>
#include <Avocado/core/error_handling.hpp>

#include <cassert>
#include <algorithm>
#include <stdexcept>

namespace
{
	using namespace avocado::nodes;
	struct Edge
	{
			Node *input;
			Node *output;
	};

	size_t get_index_of(const std::weak_ptr<Node> &node, const std::vector<std::shared_ptr<Node>> &list)
	{
		auto iter = std::find_if(list.begin(), list.end(), [node](const std::shared_ptr<Node> &x)
		{	return x ==node.lock();});
		assert(iter != list.end());
		return std::distance(list.begin(), iter);
	}

	thread_local size_t letter_counter = 0;
	const std::vector<char> letters = { 'x', 'y', 'z', 't', 'u', 'v', 'w' };
}

namespace avocado
{
	using namespace nodes;

	/*
	 * private
	 */
	node_reference Expression::add_node(std::shared_ptr<Node> newNode, std::initializer_list<node_reference> inputs)
	{
		newNode->setExpression(*this);
		m_list_of_nodes.push_back(newNode);
		for (auto iter = inputs.begin(); iter < inputs.end(); iter++)
			Node::createLink(*(iter->getNode().lock()), *newNode);
		newNode->calculateOutputShape();
		return node_reference(newNode);
	}
	/*
	 * public
	 */
	Expression::Expression() :
			m_letter(letters[letter_counter])
	{
		letter_counter = (letter_counter + 1) & letters.size();
	}
	Expression::~Expression()
	{
		letter_counter = (letter_counter - 1) & letters.size();
	}
	char Expression::debug_letter() const noexcept
	{
		return m_letter;
	}
	size_t Expression::getIndexOf(const Node &node) const
	{
		for (size_t i = 0; i < m_list_of_nodes.size(); i++)
			if (m_list_of_nodes[i].get() == &node)
				return i;
		throw std::logic_error("getIndexOf() : Node '" + node.toString() + "' is not a part of this expression");
	}
	std::weak_ptr<Node> Expression::getNodePointer(const Node &n) const
	{
		for (auto iter = m_list_of_nodes.begin(); iter < m_list_of_nodes.end(); iter++)
			if (iter->get() == &n)
				return std::weak_ptr<Node>(*iter);
		throw std::logic_error("No such node");
	}

	void Expression::replaceNode(std::shared_ptr<Node> node, Expression &e)
	{
		if (not instanceof<Input>(node.get()))
			assert(node->numberOfInputs() == e.m_inputs.size());
		if (not instanceof<Output>(node.get()))
			assert(node->numberOfOutputs() == e.m_outputs.size());

		m_list_of_nodes.insert(m_list_of_nodes.end(), e.m_list_of_nodes.begin(), e.m_list_of_nodes.end());
	}
	void Expression::removeNode(std::weak_ptr<Node> node, bool restoreLinks)
	{
	}
	Expression Expression::clone() const
	{
		Expression result;
		for (size_t i = 0; i < m_list_of_nodes.size(); i++)
		{
			std::shared_ptr<Node> new_node(m_list_of_nodes[i]->clone());
			new_node->setExpression(result);
			result.m_list_of_nodes.push_back(new_node);
		}

		for (size_t i = 0; i < m_list_of_nodes.size(); i++)
		{
			for (size_t j = 0; j < m_list_of_nodes[i]->numberOfInputs(); j++)
			{
				size_t idx = get_index_of(m_list_of_nodes[i]->getInputNodePointer(j), m_list_of_nodes);
				Node::createLink(*(result.m_list_of_nodes[idx]), *(result.m_list_of_nodes[i]));
			}
			result.m_list_of_nodes[i]->calculateOutputShape();
		}

		for (size_t i = 0; i < m_inputs.size(); i++)
			result.m_inputs.push_back(result.m_list_of_nodes[get_index_of(m_inputs[i], m_list_of_nodes)]);
		for (size_t i = 0; i < m_outputs.size(); i++)
			result.m_outputs.push_back(result.m_list_of_nodes[get_index_of(m_outputs[i], m_list_of_nodes)]);
		for (size_t i = 0; i < m_targets.size(); i++)
			result.m_targets.push_back(result.m_list_of_nodes[get_index_of(m_targets[i], m_list_of_nodes)]);
		for (size_t i = 0; i < m_losses.size(); i++)
			result.m_losses.push_back(result.m_list_of_nodes[get_index_of(m_losses[i], m_list_of_nodes)]);
		for (size_t i = 0; i < m_metrics.size(); i++)
			result.m_metrics.push_back(result.m_list_of_nodes[get_index_of(m_metrics[i], m_list_of_nodes)]);
		for (size_t i = 0; i < m_variables.size(); i++)
			result.m_variables.push_back(result.m_list_of_nodes[get_index_of(m_variables[i], m_list_of_nodes)]);
		for (size_t i = 0; i < m_trainables.size(); i++)
			result.m_trainables.push_back(result.m_list_of_nodes[get_index_of(m_trainables[i], m_list_of_nodes)]);

		return result;
	}
	void Expression::sort()
	{
		// Based on https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
		std::vector<Edge> list_of_edges;
		std::vector<std::shared_ptr<Node>> result; // L <- Empty list that will contain the sorted elements

		std::vector<std::shared_ptr<Node>> input_nodes; // S <- Set of all nodes with no incoming edge
		for (size_t i = 0; i < m_list_of_nodes.size(); i++)
			if (m_list_of_nodes[i]->numberOfInputs() == 0)
				input_nodes.push_back(m_list_of_nodes[i]);

		while (not input_nodes.empty())
		{
			std::shared_ptr<Node> n = input_nodes.front();
			input_nodes.erase(input_nodes.begin()); // remove a node n from S
			result.push_back(n); // add n to L

			while (n->numberOfOutputs() > 0) // for each node m with an edge e from n to m do (repeat until there are no more edges)
			{
				std::shared_ptr<Node> m = n->getOutputNodePointer(0).lock();
				list_of_edges.push_back(Edge { n.get(), m.get() });
				Node::removeLink(*n, *m); // remove edge e from the graph
				if (m->numberOfInputs() == 0) // if m has no other incoming edges then
					input_nodes.push_back(m); // insert m into S
			}
		}
		assert(m_list_of_nodes.size() == result.size());
		m_list_of_nodes = result;
		for (auto edge = list_of_edges.begin(); edge < list_of_edges.end(); edge++)
			Node::createLink(*(edge->input), *(edge->output)); // re-create all edges

//		for (size_t i = 0; i < m_list_of_nodes.size(); i++)
//			m_list_of_nodes[i]->setIndex(i);
	}
	void Expression::invert()
	{
		std::vector<Edge> list_of_edges;
		for (size_t i = 0; i < m_list_of_nodes.size(); i++)
		{
			std::shared_ptr<Node> n = m_list_of_nodes[i];
			while (n->numberOfOutputs() > 0)
			{
				std::shared_ptr<Node> m = n->getOutputNodePointer(0).lock();
				list_of_edges.push_back(Edge { n.get(), m.get() });
				Node::removeLink(*n, *m);
			}
		}
		for (auto edge = list_of_edges.begin(); edge < list_of_edges.end(); edge++)
			Node::createLink(*(edge->input), *(edge->output));

		sort();
		std::swap(m_inputs, m_outputs);

		std::cout << '\n';
		for (size_t i = 0; i < m_list_of_nodes.size(); i++)
		{
			std::cout << m_list_of_nodes[i]->text() << " : inputs [";
			for (size_t j = 0; j < m_list_of_nodes[i]->numberOfInputs(); j++)
				std::cout << m_list_of_nodes[i]->getInput(j).text() << ", ";
			std::cout << "] : outputs [";
			for (size_t j = 0; j < m_list_of_nodes[i]->numberOfOutputs(); j++)
				std::cout << m_list_of_nodes[i]->getOutput(j).text() << ", ";
			std::cout << "]\n";
		}
	}
	Expression Expression::getBackprop()
	{
		// first sort the nodes in inverted order
		std::vector<Edge> list_of_edges;
		std::vector<size_t> ordering;

		std::vector<std::shared_ptr<Node>> output_nodes;
		for (size_t i = 0; i < m_list_of_nodes.size(); i++)
			if (m_list_of_nodes[i]->numberOfOutputs() == 0)
				output_nodes.push_back(m_list_of_nodes[i]);

		while (not output_nodes.empty())
		{
			std::shared_ptr<Node> n = output_nodes.front();
			output_nodes.erase(output_nodes.begin());
			ordering.push_back(getIndexOf(*n));

			while (n->numberOfInputs() > 0)
			{
				std::shared_ptr<Node> m = n->getInputNodePointer(0).lock();
				list_of_edges.push_back(Edge { m.get(), n.get() });
				Node::removeLink(*m, *n);
				if (m->numberOfOutputs() == 0)
					output_nodes.push_back(m);
			}
		}
		// re-create all edges
		for (auto edge = list_of_edges.begin(); edge < list_of_edges.end(); edge++)
			Node::createLink(*(edge->input), *(edge->output));

		Expression result;
		std::vector<std::vector<node_reference>> partial_outputs(ordering.size());
		for (size_t i = 0; i < ordering.size(); i++)
		{
			std::shared_ptr<Node> node = m_list_of_nodes[ordering[i]];
			std::vector<node_reference> gradients;
			for (size_t j = 0; j < node->numberOfOutputs(); j++)
			{
				const Node &out = node->getOutput(j);
				if (&(out.getExpression()) == this) // exclude nodes that are not part of this expression
				{
					size_t idx = 0;
					for (size_t k = 0; k < out.numberOfInputs(); k++)
						if (&(out.getInput(k)) == node.get())
						{
							idx = k;
							break;
						}
					node_reference grad(partial_outputs.at(getIndexOf(out)).at(idx));
					gradients.push_back(grad);
				}
			}
			partial_outputs[ordering[i]] = node->getBackprop(result, gradients);
		}

		return result;
	}

	std::string Expression::toString() const
	{
		std::string result;
		for (size_t i = 0; i < m_list_of_nodes.size(); i++)
			result += m_list_of_nodes[i]->toString() + '\n';
		return result;
	}

	node_reference Expression::input(const Shape &shape)
	{
		std::shared_ptr<Node> tmp = std::make_shared<Input>(shape);
		m_inputs.push_back(std::weak_ptr<Node>(tmp));
		return add_node(tmp, { });
	}
	void Expression::output(const node_reference &a)
	{
		std::shared_ptr<Node> tmp = std::make_shared<Output>();
		m_outputs.push_back(std::weak_ptr<Node>(tmp));
		add_node(tmp, { a });
	}
	node_reference Expression::target(const node_reference &a)
	{
		std::shared_ptr<Node> tmp = std::make_shared<Target>(a.getNode().lock()->getOutputShape());
		m_targets.push_back(std::weak_ptr<Node>(tmp));
		return add_node(tmp, { });
	}
	void Expression::loss(const node_reference &x)
	{
		std::shared_ptr<Node> tmp = std::make_shared<Loss>();
		m_losses.push_back(std::weak_ptr<Node>(tmp));
		add_node(tmp, { x });
	}
	void Expression::metric(const node_reference &x)
	{
		std::shared_ptr<Node> tmp = std::make_shared<Metric>();
		m_metrics.push_back(std::weak_ptr<Node>(tmp));
		add_node(tmp, { x });
	}
	node_reference Expression::variable(const Shape &shape)
	{
		std::shared_ptr<Node> tmp = std::make_shared<Variable>(shape);
		m_variables.push_back(std::weak_ptr<Node>(tmp));
		return add_node(tmp, { });
	}
	node_reference Expression::trainable(const Shape &shape)
	{
		std::shared_ptr<Node> tmp = std::make_shared<Trainable>(shape);
		m_trainables.push_back(std::weak_ptr<Node>(tmp));
		return add_node(tmp, { });
	}

	node_reference Expression::view(const node_reference &a)
	{
		return add_node(std::make_shared<View>(), { a });
	}
	node_reference Expression::view(const Node *x)
	{
		for (auto iter = x->getExpression().m_list_of_nodes.begin(); iter < x->getExpression().m_list_of_nodes.end(); iter++)
			if (iter->get() == x)
				return add_node(std::make_shared<View>(), { node_reference(*iter) });
		throw std::logic_error("Expression::view() : node is not a part of this expression");
	}
	node_reference Expression::view(const std::weak_ptr<nodes::Node> &x)
	{
		return add_node(std::make_shared<View>(), { node_reference(x) });
	}
	node_reference Expression::identity(const node_reference &a)
	{
		return add_node(std::make_shared<Identity>(), { a });
	}
	node_reference Expression::one()
	{
		return add_node(std::make_shared<One>(), { });
	}
	node_reference Expression::zero()
	{
		return add_node(std::make_shared<Zero>(), { });
	}
	node_reference Expression::constant(double value)
	{
		return add_node(std::make_shared<Constant>(value), { });
	}
	/*
	 * Arithmetic operators.
	 */
	node_reference Expression::neg(const node_reference &a)
	{
		return add_node(std::make_shared<Negation>(), { a });
	}
	node_reference Expression::add(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<Addition>(), { a, b });
	}
	node_reference Expression::sub(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<Subtraction>(), { a, b });
	}
	node_reference Expression::mul(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<Multiplication>(), { a, b });
	}
	node_reference Expression::div(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<Division>(), { a, b });
	}
	node_reference Expression::mod(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<Modulo>(), { a, b });
	}
//	/*
//	 * Bitwise operators
//	 */
	node_reference Expression::bitwise_not(const node_reference &a)
	{
		return add_node(std::make_shared<BitwiseNot>(), { a });
	}
	node_reference Expression::bitwise_and(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<BitwiseAnd>(), { a, b });
	}
	node_reference Expression::bitwise_or(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<BitwiseOr>(), { a, b });
	}
	node_reference Expression::bitwise_xor(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<BitwiseXor>(), { a, b });
	}
	node_reference Expression::bitwise_shift_left(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<BitwiseShiftLeft>(), { a, b });
	}
	node_reference Expression::bitwise_shift_right(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<BitwiseShiftRight>(), { a, b });
	}
//	/*
//	 * Comparison operators.
//	 */
	node_reference Expression::equal(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<Equal>(), { a, b });
	}
	node_reference Expression::not_equal(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<NotEqual>(), { a, b });
	}
	node_reference Expression::lower_than(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<LowerThan>(), { a, b });
	}
	node_reference Expression::greater_than(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<GreaterThan>(), { a, b });
	}
	node_reference Expression::lower_or_equal(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<LowerOrEqual>(), { a, b });
	}
	node_reference Expression::greater_or_equal(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<GreaterOrEqual>(), { a, b });
	}
//	/*
//	 * Arithmetic functions
//	 */
	node_reference Expression::abs(const node_reference &a)
	{
		return add_node(std::make_shared<AbsoluteValue>(), { a });
	}
	node_reference Expression::sign(const node_reference &a)
	{
		return add_node(std::make_shared<Sign>(), { a });
	}
	node_reference Expression::floor(const node_reference &a)
	{
		return add_node(std::make_shared<Floor>(), { a });
	}
	node_reference Expression::ceil(const node_reference &a)
	{
		return add_node(std::make_shared<Ceil>(), { a });
	}
	node_reference Expression::square(const node_reference &a)
	{
		return add_node(std::make_shared<Square>(), { a });
	}
	node_reference Expression::cube(const node_reference &a)
	{
		return add_node(std::make_shared<Cube>(), { a });
	}
	node_reference Expression::pow(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<Power>(), { a, b });
	}
	node_reference Expression::sqrt(const node_reference &a)
	{
		return add_node(std::make_shared<SquareRoot>(), { a });
	}
	node_reference Expression::cbrt(const node_reference &a)
	{
		return add_node(std::make_shared<CubeRoot>(), { a });
	}
	node_reference Expression::sin(const node_reference &a)
	{
		return add_node(std::make_shared<Sine>(), { a });
	}
	node_reference Expression::cos(const node_reference &a)
	{
		return add_node(std::make_shared<Cosine>(), { a });
	}
	node_reference Expression::tan(const node_reference &a)
	{
		return add_node(std::make_shared<Tangent>(), { a });
	}
	node_reference Expression::sinh(const node_reference &a)
	{
		return add_node(std::make_shared<HyperbolicalSine>(), { a });
	}
	node_reference Expression::cosh(const node_reference &a)
	{
		return add_node(std::make_shared<HyperbolicalCosine>(), { a });
	}
	node_reference Expression::tanh(const node_reference &a)
	{
		return add_node(std::make_shared<HyperbolicalTangent>(), { a });
	}
	node_reference Expression::exp(const node_reference &a)
	{
		return add_node(std::make_shared<Exponential>(), { a });
	}
	node_reference Expression::exp2(const node_reference &a)
	{
		return add_node(std::make_shared<Exponential2>(), { a });
	}
	node_reference Expression::log(const node_reference &a)
	{
		return add_node(std::make_shared<LogarithmNatural>(), { a });
	}
	node_reference Expression::log10(const node_reference &a)
	{
		return add_node(std::make_shared<LogarithmBase10>(), { a });
	}
	node_reference Expression::log2(const node_reference &a)
	{
		return add_node(std::make_shared<LogarithmBase2>(), { a });
	}

	node_reference Expression::min(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<Minimum>(), { a, b });
	}
	node_reference Expression::max(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<Maximum>(), { a, b });
	}
//	/*
//	 * Special functions
//	 */
	node_reference Expression::select(const node_reference &a, const node_reference &b, const node_reference &c)
	{
		return add_node(std::make_shared<Select>(), { a, b, c });
	}
//	/*
//	 * Activation functions
//	 */
	node_reference Expression::sigmoid(const node_reference &a)
	{
		return add_node(std::make_shared<Sigmoid>(), { a });
	}
	node_reference Expression::relu(const node_reference &a)
	{
		return add_node(std::make_shared<ReLU>(), { a });
	}
	/*
	 * Reduction
	 */
	node_reference Expression::reduce_add(const node_reference &a, std::initializer_list<int> axes)
	{
		return add_node(std::make_shared<ReduceAdd>(axes), { a });
	}
	node_reference Expression::reduce_mul(const node_reference &a, std::initializer_list<int> axes)
	{
		return add_node(std::make_shared<ReduceMul>(axes), { a });
	}
	node_reference Expression::reduce_min(const node_reference &a, std::initializer_list<int> axes)
	{
		return add_node(std::make_shared<ReduceMin>(axes), { a });
	}
	node_reference Expression::reduce_max(const node_reference &a, std::initializer_list<int> axes)
	{
		return add_node(std::make_shared<ReduceMax>(axes), { a });
	}
	node_reference Expression::reduce_and(const node_reference &a, std::initializer_list<int> axes)
	{
		return add_node(std::make_shared<ReduceAnd>(axes), { a });
	}
	node_reference Expression::reduce_or(const node_reference &a, std::initializer_list<int> axes)
	{
		return add_node(std::make_shared<ReduceOr>(axes), { a });
	}
	/*
	 * Compound
	 */
	node_reference Expression::transpose(const node_reference &a, const std::vector<int> &order)
	{
		return add_node(std::make_shared<Transpose>(order), { a });
	}
	node_reference Expression::matmul(const node_reference &a, const node_reference &b, char opA, char opB)
	{
		return add_node(std::make_shared<MatrixMultiplication>(opA, opB), { a, b });
	}
	node_reference Expression::conv(const node_reference &a, const node_reference &b, std::initializer_list<int> filterShape)
	{
		return add_node(std::make_shared<Convolution>(filterShape), { a, b });
	}

	ExpressionTopologyError::ExpressionTopologyError(const char *function, const std::string &comment) :
			std::logic_error(std::string(function) + " : " + comment)
	{
	}

} /* namespace avocado */

