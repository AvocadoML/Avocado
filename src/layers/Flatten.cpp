/*
 * Flatten.cpp
 *
 *  Created on: Feb 19, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/layers/Flatten.hpp>
#include <Avocado/core/Context.hpp>
#include <Avocado/core/Scalar.hpp>
#include <Avocado/utils/json.hpp>
#include <Avocado/math/activations.hpp>
#include <Avocado/math/tensor_operations.hpp>

#include <Avocado/utils/static_block.hpp>

namespace avocado
{
	static_block
	{
//		registerLayer(Flatten());
	}

	void Flatten::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "Flatten layer expects single input shape");
		m_input_shapes = shapes;
	}
	Shape Flatten::getOutputShape() const
	{
		if (m_input_shapes.size() != 1)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		return Shape( { getInputShape().firstDim(), getInputShape().volumeWithoutFirstDim() });
	}

	std::string Flatten::name() const
	{
		return "Flatten";
	}

	Flatten* Flatten::clone(const Json &config) const
	{
		std::unique_ptr<Flatten> result = std::make_unique<Flatten>();
		result->m_nonlinearity = nonlinearityFromString(config["nonlinearity"]);
		return result.release();
	}

	void Flatten::forward(const std::vector<Tensor> &input, Tensor &output, Scalar alpha, Scalar beta)
	{
		assert(input.size() == 1);
		assert(same_device(context(), input[0], output));

		math::copyTensor(context(), output, input[0]);
		math::activationForwardInPlace(context(), m_nonlinearity, output);
	}
	void Flatten::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradientIn, Tensor &gradientOut, Scalar alpha, Scalar beta)
	{
		assert(input.size() == 1);
		assert(gradientIn.size() == 1);

		math::activationBackwardInPlace(context(), m_nonlinearity, output, gradientOut);
		math::copyTensor(context(), gradientIn[0], gradientOut);
	}

} /* namespace avocado */

