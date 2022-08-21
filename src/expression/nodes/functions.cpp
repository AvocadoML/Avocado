/*
 * functions.cpp
 *
 *  Created on: Jul 31, 2022
 *      Author: maciek
 */

#include <Avocado/expression/nodes/functions.hpp>
#include <Avocado/expression/Expression.hpp>
#include <Avocado/core/error_handling.hpp>

#include <cassert>

namespace avocado
{
	namespace nodes
	{

		AbsoluteValue* AbsoluteValue::clone() const
		{
			return new AbsoluteValue();
		}
		std::string AbsoluteValue::toString() const
		{
			return this->text() + " = abs(" + getInput(0).text() + ")";
		}
		std::vector<node_reference> AbsoluteValue::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto x = e.view(m_inputs.at(0));
			auto dx = dy * e.sign(x);
			return std::vector<node_reference>( { dx });
		}

		Sign* Sign::clone() const
		{
			return new Sign();
		}
		std::string Sign::toString() const
		{
			return this->text() + " = sign(" + getInput(0).text() + ")";
		}
		std::vector<node_reference> Sign::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto dx = e.zero();
			return std::vector<node_reference>( { dx });
		}

		Floor* Floor::clone() const
		{
			return new Floor();
		}
		std::string Floor::toString() const
		{
			return this->text() + " = floor(" + getInput(0).text() + ")";
		}
		std::vector<node_reference> Floor::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto dx = e.zero();
			return std::vector<node_reference>( { dx });
		}

		Ceil* Ceil::clone() const
		{
			return new Ceil();
		}
		std::string Ceil::toString() const
		{
			return this->text() + " = ceil(" + getInput(0).text() + ")";
		}
		std::vector<node_reference> Ceil::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto dx = e.zero();
			return std::vector<node_reference>( { dx });
		}

		/*
		 * Power
		 */
		Square* Square::clone() const
		{
			return new Square();
		}
		std::string Square::toString() const
		{
			return this->text() + " = square(" + getInput(0).text() + ")";
		}
		std::vector<node_reference> Square::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto x = e.view(m_inputs.at(0));
			auto dx = dy * e.constant(2) * x;
			return std::vector<node_reference>( { dx });
		}

		Cube* Cube::clone() const
		{
			return new Cube();
		}
		std::string Cube::toString() const
		{
			return this->text() + " = cube(" + getInput(0).text() + ")";
		}
		std::vector<node_reference> Cube::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto x = e.view(m_inputs.at(0));
			auto dx = dy * e.constant(3) * e.square(x);
			return std::vector<node_reference>( { dx });
		}

		Power* Power::clone() const
		{
			return new Power();
		}
		std::string Power::toString() const
		{
			return this->text() + " = pow(" + getInput(0).text() + ", " + getInput(1).text() + ")";
		}
		std::vector<node_reference> Power::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto x1 = e.view(m_inputs.at(0));
			auto x2 = e.view(m_inputs.at(1));
			auto dx1 = dy * x2 * e.pow(x1, x2 - e.one());
			auto dx2 = dy * e.pow(x1, x2) * e.log(x1);
			return std::vector<node_reference>( { dx1, dx2 });
		}

		SquareRoot* SquareRoot::clone() const
		{
			return new SquareRoot();
		}
		std::string SquareRoot::toString() const
		{
			return this->text() + " = sqrt(" + getInput(0).text() + ")";
		}
		std::vector<node_reference> SquareRoot::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto x = e.view(m_inputs.at(0));
			auto dx = dy / (e.constant(2) * e.sqrt(x));
			return std::vector<node_reference>( { dx });
		}

		CubeRoot* CubeRoot::clone() const
		{
			return new CubeRoot();
		}
		std::string CubeRoot::toString() const
		{
			return this->text() + " = cbrt(" + getInput(0).text() + ")";
		}
		std::vector<node_reference> CubeRoot::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto x = e.view(m_inputs.at(0));
			auto dx = dy / (e.constant(3) * e.cbrt(e.square(x)));
			return std::vector<node_reference>( { dx });
		}

		/*
		 * Trigonometrical
		 */
		Sine* Sine::clone() const
		{
			return new Sine();
		}
		std::string Sine::toString() const
		{
			return this->text() + " = sin(" + getInput(0).text() + ")";
		}
		std::vector<node_reference> Sine::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto x = e.view(m_inputs.at(0));
			auto dx = dy * e.cos(x);
			return std::vector<node_reference>( { dx });
		}

		Cosine* Cosine::clone() const
		{
			return new Cosine();
		}
		std::string Cosine::toString() const
		{
			return this->text() + " = cos(" + getInput(0).text() + ")";
		}
		std::vector<node_reference> Cosine::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto x = e.view(m_inputs.at(0));
			auto dx = -dy * e.sin(x);
			return std::vector<node_reference>( { dx });
		}

		Tangent* Tangent::clone() const
		{
			return new Tangent();
		}
		std::string Tangent::toString() const
		{
			return this->text() + " = tan(" + getInput(0).text() + ")";
		}
		std::vector<node_reference> Tangent::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto x = e.view(m_inputs.at(0));
			auto dx = dy / e.square(e.cos(x));
			return std::vector<node_reference>( { dx });
		}

		/*
		 * Hyperbolical
		 */
		HyperbolicalSine* HyperbolicalSine::clone() const
		{
			return new HyperbolicalSine();
		}
		std::string HyperbolicalSine::toString() const
		{
			return this->text() + " = sinh(" + getInput(0).text() + ")";
		}
		std::vector<node_reference> HyperbolicalSine::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto x = e.view(m_inputs.at(0));
			auto dx = dy * e.cosh(x);
			return std::vector<node_reference>( { dx });
		}

		HyperbolicalCosine* HyperbolicalCosine::clone() const
		{
			return new HyperbolicalCosine();
		}
		std::string HyperbolicalCosine::toString() const
		{
			return this->text() + " = cosh(" + getInput(0).text() + ")";
		}
		std::vector<node_reference> HyperbolicalCosine::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto x = e.view(m_inputs.at(0));
			auto dx = dy * e.sinh(x);
			return std::vector<node_reference>( { dx });
		}

		HyperbolicalTangent* HyperbolicalTangent::clone() const
		{
			return new HyperbolicalTangent();
		}
		std::string HyperbolicalTangent::toString() const
		{
			return this->text() + " = tanh(" + getInput(0).text() + ")";
		}
		std::vector<node_reference> HyperbolicalTangent::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto y = e.view(this);
			auto dx = dy * (e.one() - e.square(y));
			return std::vector<node_reference>( { dx });
		}

		/*
		 * Exponential
		 */
		Exponential* Exponential::clone() const
		{
			return new Exponential();
		}
		std::string Exponential::toString() const
		{
			return this->text() + " = exp(" + getInput(0).text() + ")";
		}
		std::vector<node_reference> Exponential::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto y = e.view(this);
			auto dx = dy * y;
			return std::vector<node_reference>( { dx });
		}

		Exponential2* Exponential2::clone() const
		{
			return new Exponential2();
		}
		std::string Exponential2::toString() const
		{
			return this->text() + " = exp2(" + getInput(0).text() + ")";
		}
		std::vector<node_reference> Exponential2::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto y = e.view(this);
			auto dx = dy * y * e.log(e.constant(2));
			return std::vector<node_reference>( { dx });
		}

		/*
		 * Logarithmic
		 */
		LogarithmNatural* LogarithmNatural::clone() const
		{
			return new LogarithmNatural();
		}
		std::string LogarithmNatural::toString() const
		{
			return this->text() + " = log(" + getInput(0).text() + ")";
		}
		std::vector<node_reference> LogarithmNatural::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto x = e.view(m_inputs.at(0));
			auto dx = dy / x;
			return std::vector<node_reference>( { dx });
		}

		LogarithmBase10* LogarithmBase10::clone() const
		{
			return new LogarithmBase10();
		}
		std::string LogarithmBase10::toString() const
		{
			return this->text() + " = log10(" + getInput(0).text() + ")";
		}
		std::vector<node_reference> LogarithmBase10::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto x = e.view(m_inputs.at(0));
			auto dx = dy / (x * e.log(e.constant(10)));
			return std::vector<node_reference>( { dx });
		}

		LogarithmBase2* LogarithmBase2::clone() const
		{
			return new LogarithmBase2();
		}
		std::string LogarithmBase2::toString() const
		{
			return this->text() + " = log2(" + getInput(0).text() + ")";
		}
		std::vector<node_reference> LogarithmBase2::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto x = e.view(m_inputs.at(0));
			auto dx = dy / (x * e.log(e.constant(2)));
			return std::vector<node_reference>( { dx });
		}

		/*
		 * Min/Max
		 */
		Minimum* Minimum::clone() const
		{
			return new Minimum();
		}
		std::string Minimum::toString() const
		{
			return this->text() + " = min(" + getInput(0).text() + ", " + getInput(0).text() + ")";
		}
		std::vector<node_reference> Minimum::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto x1 = e.view(m_inputs.at(0));
			auto x2 = e.view(m_inputs.at(1));
			auto dx1 = e.select(x1 <= x2, dy, e.zero());
			auto dx2 = e.select(x1 <= x2, e.zero(), dy);
			return std::vector<node_reference>( { dx1, dx2 });
		}

		Maximum* Maximum::clone() const
		{
			return new Maximum();
		}
		std::string Maximum::toString() const
		{
			return this->text() + " = max(" + getInput(0).text() + ", " + getInput(0).text() + ")";
		}
		std::vector<node_reference> Maximum::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto x1 = e.view(m_inputs.at(0));
			auto x2 = e.view(m_inputs.at(1));
			auto dx1 = e.select(x1 >= x2, dy, e.zero());
			auto dx2 = e.select(x1 >= x2, e.zero(), dy);
			return std::vector<node_reference>( { dx1, dx2 });
		}

	} /* namespace nodes */
} /* namespace avocado */

