/*
 * technical.cpp
 *
 *  Created on: Jul 30, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/expression/nodes/technical.hpp>
#include <Avocado/expression/Expression.hpp>
#include <Avocado/core/Shape.hpp>
#include <Avocado/core/error_handling.hpp>

namespace avocado
{
	namespace nodes
	{

		Input::Input(const Shape &shape)
		{
			m_output_shape = shape;
		}
		Input* Input::clone() const
		{
			return new Input(m_output_shape);
		}
		std::string Input::toString() const
		{
			return this->text() + " <- input";
		}

		Target::Target(const Shape &shape)
		{
			m_output_shape = shape;
		}
		Target* Target::clone() const
		{
			return new Target(m_output_shape);
		}
		std::string Target::toString() const
		{
			return this->text() + " <- target";
		}

		Output* Output::clone() const
		{
			return new Output();
		}
		void Output::calculateOutputShape()
		{
			if (numberOfInputs() != 1)
				throw ExpressionTopologyError(METHOD_NAME, "node must have exactly one input");
			m_output_shape = getInput(0).getOutputShape();
		}
		std::string Output::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " -> output";
		}

		Loss* Loss::clone() const
		{
			return new Loss();
		}
		void Loss::calculateOutputShape()
		{
			if (numberOfInputs() != 1)
				throw ExpressionTopologyError(METHOD_NAME, "node must have exactly one input");
			m_output_shape = getInput(0).getOutputShape();
		}
		std::string Loss::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " -> loss";
		}
		std::vector<node_reference> Loss::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dx = e.one();
			return std::vector<node_reference>( { dx });
		}

		Metric* Metric::clone() const
		{
			return new Metric();
		}
		void Metric::calculateOutputShape()
		{
			if (numberOfInputs() != 1)
				throw ExpressionTopologyError(METHOD_NAME, "node must have exactly one input");
			m_output_shape = getInput(0).getOutputShape();
		}
		std::string Metric::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " -> metric";
		}

		Variable::Variable(const Shape &shape)
		{
			m_output_shape = shape;
		}
		Variable* Variable::clone() const
		{
			return new Variable(m_output_shape);
		}
		std::string Variable::toString() const
		{
			return this->text() + " <- variable";
		}

		Trainable::Trainable(const Shape &shape)
		{
			m_output_shape = shape;
		}
		Trainable* Trainable::clone() const
		{
			return new Trainable(m_output_shape);
		}
		std::string Trainable::toString() const
		{
			return this->text() + " <- trainable";
		}
		std::vector<node_reference> Trainable::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			e.output(dy);
			return std::vector<node_reference>();
		}

		View* View::clone() const
		{
			return new View();
		}
		void View::calculateOutputShape()
		{
			if (numberOfInputs() != 1)
				throw ExpressionTopologyError(METHOD_NAME, "node must have exactly one input");
			m_output_shape = getInput(0).getOutputShape();
		}
		std::string View::toString() const
		{
			return this->text() + " = view of " + getInput(0).text();
		}

		Identity* Identity::clone() const
		{
			return new Identity();
		}
		std::string Identity::toString() const
		{
			return this->text() + " = " + getInput(0).text();
		}
		std::vector<node_reference> Identity::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto dx = e.identity(dy);
			return std::vector<node_reference>( { dx });
		}

		/*
		 * Constant values
		 */
		Constant::Constant(double value) :
				m_values( { value })
		{
		}
		Constant::Constant(const std::vector<double> &values) :
				m_values(values)
		{
		}
		Constant* Constant::clone() const
		{
			return new Constant(m_values);
		}
		void Constant::calculateOutputShape()
		{
			m_output_shape = { static_cast<int>(m_values.size()) };
		}
		std::string Constant::toString() const
		{
			std::string result = this->text() + " = ";
			if (m_values.size() == 1)
				result += std::to_string(m_values[0]);
			else
			{
				result += "{";
				for (size_t i = 0; i < m_values.size(); i++)
				{
					if (i != 0)
						result += ", ";
					result += std::to_string(m_values[i]);
				}
				result += "}";
			}
			return result;
		}
		size_t Constant::size() const noexcept
		{
			return m_values.size();
		}
		double Constant::getValue(size_t index) const
		{
			return m_values.at(index);
		}

		One::One() :
				Constant(1.0)
		{
		}
		One* One::clone() const
		{
			return new One();
		}
		std::string One::toString() const
		{
			return this->text() + " = 1";
		}

		Zero::Zero() :
				Constant(0.0)
		{
		}
		Zero* Zero::clone() const
		{
			return new Zero();
		}
		std::string Zero::toString() const
		{
			return this->text() + " = 0";
		}

		/*
		 * Selection
		 */
		Select* Select::clone() const
		{
			return new Select();
		}
		void Select::calculateOutputShape()
		{
			if (numberOfInputs() != 3)
				throw ExpressionTopologyError(METHOD_NAME, "node must have exactly three inputs");
			m_output_shape = getShapeAfterBroadcasting(getInput(1).getOutputShape(), getInput(2).getOutputShape());
		}
		std::string Select::toString() const
		{
			return this->text() + " = select(" + getInput(0).text() + ", " + getInput(1).text() + ", " + getInput(2).text() + ")";
		}
		std::vector<node_reference> Select::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto x = e.view(m_inputs.at(0));
			auto dx1 = e.zero();
			auto dx2 = e.select(x, dy, e.zero());
			auto dx3 = e.select(x, e.zero(), dy);
			return std::vector<node_reference>( { dx1, dx2, dx3 });
		}

	} /* namespace nodes */
} /* namespace avocado */

