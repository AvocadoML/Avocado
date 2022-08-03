/*
 * Activation.cpp
 *
 *  Created on: Feb 10, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/layers/Activation.hpp>
#include <Avocado/core/Context.hpp>
#include <Avocado/core/Shape.hpp>
#include <Avocado/core/Tensor.hpp>
#include <Avocado/core/Scalar.hpp>
#include <Avocado/utils/json.hpp>
#include <Avocado/utils/serialization.hpp>
#include <Avocado/math/activations.hpp>

#include <cassert>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include <Avocado/utils/static_block.hpp>

namespace avocado
{
	static_block
	{
//		registerLayer(Activation());
	}

	Activation::Activation(const std::string &activation) :
			Layer(activation)
	{
	}
	Activation::Activation(NonlinearityType activation) :
			Layer(toString(activation))
	{
	}

	void Activation::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "Activation layer expects single input shape");
		m_input_shapes = shapes;
	}
	Shape Activation::getOutputShape() const
	{
		if (m_input_shapes.size() != 1)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		return m_input_shapes[0];
	}

	std::string Activation::name() const
	{
		return "Activation";
	}

	Activation* Activation::clone(const Json &config) const
	{
		return new Activation(config["nonlinearity"]);
	}

	void Activation::forward(const std::vector<Tensor> &input, Tensor &output, Scalar alpha, Scalar beta)
	{
		assert(input.size() == 1);
		math::activationForward(context(), m_nonlinearity, 1, input[0], 0, output);
	}
	void Activation::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradientIn, Tensor &gradientOut,
			Scalar alpha, Scalar beta)
	{
		assert(input.size() == 1);
		assert(gradientIn.size() == 1);
		math::activationBackward(context(), m_nonlinearity, 1, output, gradientOut, beta, gradientIn[0]);
	}

} /* namespace avocado */

