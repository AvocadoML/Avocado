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
		Expression AbsoluteValue::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto x = result.view(m_inputs.at(0));
			result.output(dy * result.sign(x));
			return result;
		}

		Sign* Sign::clone() const
		{
			return new Sign();
		}
		std::string Sign::toString() const
		{
			return this->text() + " = sign(" + getInput(0).text() + ")";
		}
		Expression Sign::getBackprop() const
		{
			Expression result;
			result.output(result.zero());
			return result;
		}

		Floor* Floor::clone() const
		{
			return new Floor();
		}
		std::string Floor::toString() const
		{
			return this->text() + " = floor(" + getInput(0).text() + ")";
		}
		Expression Floor::getBackprop() const
		{
			Expression result;
			result.output(result.zero());
			return result;
		}

		Ceil* Ceil::clone() const
		{
			return new Ceil();
		}
		std::string Ceil::toString() const
		{
			return this->text() + " = ceil(" + getInput(0).text() + ")";
		}
		Expression Ceil::getBackprop() const
		{
			Expression result;
			result.output(result.zero());
			return result;
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
		Expression Square::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto x = result.view(m_inputs.at(0));
			result.output(dy * result.constant(2) * x);
			return result;
		}

		Cube* Cube::clone() const
		{
			return new Cube();
		}
		std::string Cube::toString() const
		{
			return this->text() + " = cube(" + getInput(0).text() + ")";
		}
		Expression Cube::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto x = result.view(m_inputs.at(0));
			result.output(dy * result.constant(3) * result.square(x));
			return result;
		}

		Power* Power::clone() const
		{
			return new Power();
		}
		std::string Power::toString() const
		{
			return this->text() + " = pow(" + getInput(0).text() + ", " + getInput(1).text() + ")";
		}
		Expression Power::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto a = result.view(m_inputs.at(0));
			auto b = result.view(m_inputs.at(1));
			result.output(dy * b * result.pow(a, b - result.one()));
			result.output(dy * result.pow(a, b) * result.log(a));
			return result;
		}

		SquareRoot* SquareRoot::clone() const
		{
			return new SquareRoot();
		}
		std::string SquareRoot::toString() const
		{
			return this->text() + " = sqrt(" + getInput(0).text() + ")";
		}
		Expression SquareRoot::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto x = result.view(m_inputs.at(0));
			result.output(dy / (result.constant(2) * result.sqrt(x)));
			return result;
		}

		CubeRoot* CubeRoot::clone() const
		{
			return new CubeRoot();
		}
		std::string CubeRoot::toString() const
		{
			return this->text() + " = cbrt(" + getInput(0).text() + ")";
		}
		Expression CubeRoot::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto x = result.view(m_inputs.at(0));
			result.output(dy / (result.constant(3) * result.cbrt(result.square(x))));
			return result;
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
		Expression Sine::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto x = result.view(m_inputs.at(0));
			result.output(dy * result.cos(x));
			return result;
		}

		Cosine* Cosine::clone() const
		{
			return new Cosine();
		}
		std::string Cosine::toString() const
		{
			return this->text() + " = cos(" + getInput(0).text() + ")";
		}
		Expression Cosine::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto x = result.view(m_inputs.at(0));
			result.output(-dy * result.sin(x));
			return result;
		}

		Tangent* Tangent::clone() const
		{
			return new Tangent();
		}
		std::string Tangent::toString() const
		{
			return this->text() + " = tan(" + getInput(0).text() + ")";
		}
		Expression Tangent::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto x = result.view(m_inputs.at(0));
			result.output(dy / result.square(result.cos(x)));
			return result;
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
		Expression HyperbolicalSine::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto x = result.view(m_inputs.at(0));
			result.output(dy * result.cosh(x));
			return result;
		}

		HyperbolicalCosine* HyperbolicalCosine::clone() const
		{
			return new HyperbolicalCosine();
		}
		std::string HyperbolicalCosine::toString() const
		{
			return this->text() + " = cosh(" + getInput(0).text() + ")";
		}
		Expression HyperbolicalCosine::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto x = result.view(m_inputs.at(0));
			result.output(dy * result.sinh(x));
			return result;
		}

		HyperbolicalTangent* HyperbolicalTangent::clone() const
		{
			return new HyperbolicalTangent();
		}
		std::string HyperbolicalTangent::toString() const
		{
			return this->text() + " = tanh(" + getInput(0).text() + ")";
		}
		Expression HyperbolicalTangent::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto y = result.view(this);
			result.output(dy * (result.one() - y) * (result.one() + y));
			return result;
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
		Expression Exponential::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto y = result.view(this);
			result.output(dy * y);
			return result;
		}

		Exponential2* Exponential2::clone() const
		{
			return new Exponential2();
		}
		std::string Exponential2::toString() const
		{
			return this->text() + " = exp2(" + getInput(0).text() + ")";
		}
		Expression Exponential2::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto y = result.view(this);
			result.output(dy * y * result.log(result.constant(2)));
			return result;
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
		Expression LogarithmNatural::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto x = result.view(m_inputs.at(0));
			result.output(dy / x);
			return result;
		}

		LogarithmBase10* LogarithmBase10::clone() const
		{
			return new LogarithmBase10();
		}
		std::string LogarithmBase10::toString() const
		{
			return this->text() + " = log10(" + getInput(0).text() + ")";
		}
		Expression LogarithmBase10::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto x = result.view(m_inputs.at(0));
			result.output(dy / (x * result.log(result.constant(10))));
			return result;
		}

		LogarithmBase2* LogarithmBase2::clone() const
		{
			return new LogarithmBase2();
		}
		std::string LogarithmBase2::toString() const
		{
			return this->text() + " = log2(" + getInput(0).text() + ")";
		}
		Expression LogarithmBase2::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto x = result.view(m_inputs.at(0));
			result.output(dy / (x * result.log(result.constant(10))));
			return result;
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
		Expression Minimum::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto x1 = result.view(m_inputs.at(0));
			auto x2 = result.view(m_inputs.at(1));
			result.output(result.select(x1 <= x2, dy, result.zero()));
			result.output(result.select(x1 <= x2, result.zero(), dy));
			return result;
		}

		Maximum* Maximum::clone() const
		{
			return new Maximum();
		}
		std::string Maximum::toString() const
		{
			return this->text() + " = max(" + getInput(0).text() + ", " + getInput(0).text() + ")";
		}
		Expression Maximum::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto x1 = result.view(m_inputs.at(0));
			auto x2 = result.view(m_inputs.at(1));
			result.output(result.select(x1 >= x2, dy, result.zero()));
			result.output(result.select(x1 >= x2, result.zero(), dy));
			return result;
		}

	} /* namespace nodes */
} /* namespace avocado */

