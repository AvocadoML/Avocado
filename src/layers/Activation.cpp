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

	void Activation::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);
		assert(same_device(context(), input[0], output));
		math::activationForward(context(), m_nonlinearity, 1, 1, 0, input[0], output);
	}
	void Activation::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next)
	{
		assert(input.size() == 1);
		assert(gradient_prev.size() == 1);
		assert(same_device(context(), input[0], output, gradient_prev[0], gradient_next));
		math::activationBackward(context(), m_nonlinearity, 1, 0, gradient_prev[0], gradient_next, output);
	}

} /* namespace avocado */

