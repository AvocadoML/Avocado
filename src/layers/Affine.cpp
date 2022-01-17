/*
 * Affine.cpp
 *
 *  Created on: Feb 19, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/layers/Affine.hpp>
#include <Avocado/core/Context.hpp>
#include <Avocado/core/Scalar.hpp>
#include <Avocado/utils/json.hpp>
#include <Avocado/math/batchnorm.hpp>
#include <Avocado/math/activations.hpp>
#include <Avocado/math/tensor_operations.hpp>

#include <Avocado/utils/static_block.hpp>

namespace avocado
{
	static_block
	{
//		registerLayer(Affine());
	}

	Affine::Affine(const std::string &activation, bool useWeights, bool useBias) :
			Layer(activation),
			m_use_weights(useWeights),
			m_use_bias(useBias)
	{
	}

	Layer& Affine::useWeights(bool b) noexcept
	{
		if (m_use_weights == false && b == true)
			m_weights = nullptr;
		m_use_weights = b;
		return *this;
	}
	bool Affine::isUsingWeights() const noexcept
	{
		return m_use_weights;
	}
	Layer& Affine::useBias(bool b) noexcept
	{
		if (m_use_bias == false && b == true)
			m_bias = nullptr;
		m_use_bias = b;
		return *this;
	}
	bool Affine::isUsingBias() const noexcept
	{
		return m_use_bias;
	}

	void Affine::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "Affine layer expects single input shape");
		if (m_input_shapes.size() != 0 && getInputShape().lastDim() != shapes[0].lastDim())
			throw ShapeMismatch(METHOD_NAME, getInputShape(), shapes[0]);

		m_input_shapes = shapes;
	}
	Shape Affine::getOutputShape() const
	{
		if (m_input_shapes.size() != 1)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		return getInputShape();
	}
	Shape Affine::getWeightShape() const
	{
		if (m_use_weights)
			return Shape( { getInputShape().lastDim() });
		else
			return Shape();
	}
	Shape Affine::getBiasShape() const
	{
		if (m_use_bias)
			return Shape( { getInputShape().lastDim() });
		else
			return Shape();
	}

	std::string Affine::name() const
	{
		return "Affine";
	}
	Json Affine::getConfig() const
	{
		Json result = Layer::getConfig();
		result["use_weights"] = m_use_weights;
		result["use_bias"] = m_use_bias;
		return result;
	}

	Affine* Affine::clone(const Json &config) const
	{
		return new Affine(config["nonlinearity"], config["use_weights"], config["use_bias"]); // @suppress("Ambiguous problem")
	}

	void Affine::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);
		assert(same_device(context(), input[0], output));

		math::affineForward(context(), 1, 0, input[0], output, getWeights().getParam(), getBias().getParam(), m_nonlinearity);
	}
	void Affine::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next)
	{
		assert(input.size() == 1);
		assert(gradient_prev.size() == 1);
		assert(same_device(context(), input[0], output, gradient_prev[0], gradient_next));

		math::activationBackward(context(), m_nonlinearity, 1, 0, gradient_next, gradient_next, output);
		if (m_use_weights)
			math::affineForward(context(), 1, 0, gradient_prev[0], gradient_next, getWeights().getParam(), Tensor( { }, dtype(), device()),
					NonlinearityType::LINEAR);
		else
			math::copyTensor(context(), gradient_prev[0], gradient_next);
		if (m_use_bias)
			math::reduceTensor(context(), math::TensorReduceOp::ADD, 1, 1, gradient_next, getBias().getUpdate());
	}
} /* namespace avocado */
