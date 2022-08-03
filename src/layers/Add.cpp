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

	void Add::forward(const std::vector<Tensor> &input, Tensor &output, Scalar alpha, Scalar beta)
	{
		assert(input.size() == m_input_shapes.size());

		math::tensorBinaryOp(context(), TensorBinaryOp::ADD, 1, input[0], 1, input[1], 0, output);
		for (size_t i = 2; i < input.size(); i++)
			math::addTensors(context(), output, input[i], 1, 1);
		math::activationForwardInPlace(context(), m_nonlinearity, output);
	}
	void Add::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradientIn, Tensor &gradientOut, Scalar alpha,
			Scalar beta)
	{
		assert(input.size() == m_input_shapes.size());
		assert(gradientIn.size() == m_input_shapes.size());

		math::activationBackwardInPlace(context(), m_nonlinearity, output, gradientOut);
		for (size_t i = 0; i < gradientIn.size(); i++)
			math::addTensors(context(), gradientIn[i], gradientOut, 1, beta);
	}
} /* namespace avocado */

