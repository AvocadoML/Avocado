/*
 * Concatenate.cpp
 *
 *  Created on: Feb 24, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/layers/Concatenate.hpp>
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
//		registerLayer(Concatenate());
	}

	void Concatenate::setInputShape(const std::vector<Shape> &shapes)
	{
		for (size_t i = 1; i < shapes.size(); i++)
		{
			if (shapes[0].length() != shapes[i].length())
				throw ShapeMismatch(METHOD_NAME, shapes[0], shapes[i]);
			for (int j = 0; j < shapes[i].length() - 1; j++)
				if (shapes[0][j] != shapes[i][j])
					throw ShapeMismatch(METHOD_NAME, shapes[0], shapes[i]);
		}
		m_input_shapes = shapes;
	}
	Shape Concatenate::getOutputShape() const
	{
		int tmp = 0;
		for (int i = 0; i < numberOfInputs(); i++)
			tmp += getInputShape(i).lastDim();
		Shape result = getInputShape();
		result[result.length() - 1] = tmp;
		return result;
	}

	std::string Concatenate::name() const
	{
		return "Concatenate";
	}

	Concatenate* Concatenate::clone(const Json &config) const
	{
		std::unique_ptr<Concatenate> result = std::make_unique<Concatenate>();
		result->m_nonlinearity = nonlinearityFromString(config["nonlinearity"]);
		return result.release();
	}

	void Concatenate::forward(const std::vector<Tensor> &input, Tensor &output, Scalar alpha, Scalar beta)
	{
		math::concatTensors(context(), output, input);
		math::activationForwardInPlace(context(), m_nonlinearity, output);
	}
	void Concatenate::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradientIn, Tensor &gradientOut,
			Scalar alpha, Scalar beta)
	{
		// FIXME scaling factors do not work for this layer
		math::activationBackwardInPlace(context(), m_nonlinearity, output, gradientOut);
		math::splitTensors(context(), gradientIn, gradientOut);
	}
} /* namespace avocado */

