/*
 * Dense.cpp
 *
 *  Created on: Feb 19, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/layers/Dense.hpp>
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
//		registerLayer(Dense(0));
	}

	Dense::Dense(int neurons, const std::string &activation, bool useBias) :
			Layer(activation),
			m_neurons(neurons),
			m_use_bias(useBias)
	{
	}

	Layer& Dense::useBias(bool b) noexcept
	{
		if (m_use_bias == false && b == true)
			m_bias = nullptr;
		m_use_bias = b;
		return *this;
	}
	bool Dense::isUsingBias() const noexcept
	{
		return m_use_bias;
	}

	void Dense::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "Dense layer expects single input shape");
		if (shapes[0].length() != 2)
			throw ShapeMismatch(METHOD_NAME, 2, shapes[0].length());

		if (m_input_shapes.size() != 0 && getInputShape().lastDim() != shapes[0].lastDim())
			throw ShapeMismatch(METHOD_NAME, getInputShape(), shapes[0]);

		m_input_shapes = shapes;
	}
	Shape Dense::getOutputShape() const
	{
		if (m_input_shapes.size() != 1)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		return Shape( { getInputShape().firstDim(), m_neurons });
	}
	Shape Dense::getWeightShape() const
	{
		return Shape( { m_neurons, getInputShape().lastDim() });
	}
	Shape Dense::getBiasShape() const
	{
		if (m_use_bias)
			return Shape( { m_neurons });
		else
			return Shape();
	}

	std::string Dense::name() const
	{
		return "Dense";
	}
	Json Dense::getConfig() const
	{
		Json result = Layer::getConfig();
		result["neurons"] = m_neurons;
		result["use_bias"] = m_use_bias;
		return result;
	}

	Dense* Dense::clone(const Json &config) const
	{
		return new Dense(config["neurons"], config["nonlinearity"], config["use_bias"]); // @suppress("Ambiguous problem")
	}

	void Dense::forward(const std::vector<Tensor> &input, Tensor &output, Scalar alpha, Scalar beta)
	{
		assert(input.size() == 1 || input.size() == 2);

		if (input.size() == 1)
		{
			math::gemm(context(), GemmOp::OP_N, GemmOp::OP_T, output, input[0], getWeights().getParam(), 1, 0);
			if (m_use_bias)
				math::addBias(context, alpha1, alpha2, input, bias, beta1, beta2, beta3, output, input[1], m_nonlinearity);
			else
				math::activationForwardInPlace(context, activation, output);
		}
		else
		{
			math::copyTensor(context(), output, input[1]);
			math::gemm(context(), GemmOp::OP_N, GemmOp::OP_T, output, input[0], getWeights().getParam(), 1, 1);
		}

	}
	void Dense::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradientIn, Tensor &gradientOut, Scalar alpha,
			Scalar beta)
	{
		assert(input.size() == 1);
		assert(gradientIn.size() == 1);

//		math::activationBackward(context(), m_nonlinearity, 1, 0, gradient_next, gradient_next, output);
		math::gemm(context(), GemmOp::OP_N, GemmOp::OP_N, gradientIn[0], gradientOut, getWeights().getParam(), 1, 0);
		math::gemm(context(), GemmOp::OP_T, GemmOp::OP_N, getWeights().getUpdate(), gradientOut, input[0], 1, 1);
		if (m_use_bias)
			math::reduceTensor(context(), TensorReduceOp::ADD, 1, 1, gradientOut, getBias().getUpdate());
	}

} /* namespace avocado */

