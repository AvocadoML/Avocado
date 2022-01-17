/*
 * Add.cpp
 *
 *  Created on: Feb 24, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/layers/Add.hpp>
#include <Avocado/core/Context.hpp>
#include <Avocado/core/Scalar.hpp>
#include <Avocado/utils/json.hpp>
#include <Avocado/math/tensor_operations.hpp>
#include <Avocado/math/activations.hpp>

#include <Avocado/utils/static_block.hpp>

namespace avocado
{
	static_block
	{
//		registerLayer(Add());
	}

	Add::Add(const std::string &activation) :
			Layer(activation)
	{
	}

	void Add::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() < 2)
			throw LogicError(METHOD_NAME, "Add layer expects at least two inputs");

		for (size_t i = 1; i < shapes.size(); i++)
			if (shapes[0] != shapes[i])
				throw ShapeMismatch(METHOD_NAME, shapes[0], shapes[i]);
		m_input_shapes = shapes;
	}
	Shape Add::getOutputShape() const
	{
		if (m_input_shapes.size() == 0)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		return getInputShape();
	}

	std::string Add::name() const
	{
		return "Add";
	}

	Add* Add::clone(const Json &config) const
	{
		return new Add(config["nonlinearity"]);
	}

	void Add::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == m_input_shapes.size());
		assert(same_device(context(), input[0], output));

		math::copyTensor(context(), output, input[0]); // copy first input tensor to output

		for (size_t i = 1; i < input.size() - 1; i++) // add all but first and last input tensors to output
			math::addTensors(context(), 1, 1, input[i], output, NonlinearityType::LINEAR);

		math::addTensors(context(), 1, 1, input.back(), output, m_nonlinearity); // add last input tensor to output and apply activation
	}
	void Add::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next)
	{
		assert(input.size() == m_input_shapes.size());
		assert(gradient_prev.size() == m_input_shapes.size());
		assert(same_device(context(), input[0], output, gradient_prev[0], gradient_next));

		math::activationBackward(context(), m_nonlinearity, 1, 0, gradient_next, gradient_next, output); // in place activation backward pass
		for (size_t i = 0; i < gradient_prev.size(); i++)
			math::copyTensor(context(), gradient_prev[i], gradient_next);
	}
} /* namespace avocado */

