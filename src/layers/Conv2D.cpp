/*
 * Conv2D.cpp
 *
 *  Created on: Feb 26, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/layers/Conv2D.hpp>
#include <Avocado/core/Context.hpp>
#include <Avocado/core/Tensor.hpp>
#include <Avocado/utils/json.hpp>

//#include <Avocado/math/conv2D_util.hpp>
//#include <Avocado/math/gemms.hpp>
//#include <Avocado/math/winograd_conv2D.hpp>
//#include <Avocado/math/tensor_operations.hpp>
#include <Avocado/math/activations.hpp>
#include <Avocado/utils/static_block.hpp>

namespace avocado
{
	static_block
	{
//		registerLayer(Conv2D(0, 0));
	}

	Conv2D::Conv2D(int filters, int kernelSize, const std::string &activation, bool useBias) :
			Conv2D(filters, { kernelSize, kernelSize }, activation, useBias)
	{
	}
	Conv2D::Conv2D(int filters, std::initializer_list<int> kernelSize, const std::string &activation, bool useBias) :
			Layer(activation)
	{
		m_output_filters = filters;
		m_config.activation = m_nonlinearity;
		std::copy(kernelSize.begin(), kernelSize.end(), m_config.kernel.begin());
		m_use_bias = useBias;
	}

	Layer& Conv2D::useBias(bool b) noexcept
	{
		if (m_use_bias == false && b == true)
			m_bias = nullptr;
		m_use_bias = b;
		return *this;
	}
	Layer& Conv2D::setPadding(ConvPadding padding) noexcept
	{
		m_padding = padding;
		return *this;
	}
	Layer& Conv2D::setStride(int stride) noexcept
	{
		return setStride( { stride, stride });
	}
	Layer& Conv2D::setStride(std::initializer_list<int> stride) noexcept
	{
		std::copy(stride.begin(), stride.end(), m_config.stride.begin());
		return *this;
	}
	Layer& Conv2D::setDilation(int dilation) noexcept
	{
		return setDilation( { dilation, dilation });
	}
	Layer& Conv2D::setDilation(std::initializer_list<int> dilation) noexcept
	{
		std::copy(dilation.begin(), dilation.end(), m_config.dilation.begin());
		return *this;
	}
	Layer& Conv2D::setGroups(int groups) noexcept
	{
		m_config.groups = groups;
		return *this;
	}
	bool Conv2D::isUsingBias() const noexcept
	{
		return m_use_bias;
	}

	void Conv2D::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1 && shapes.size() != 2)
			throw IllegalArgument(METHOD_NAME, "Conv2D layer expects either one or two input shapes");
		if (shapes[0].length() != 4)
			throw IllegalArgument(METHOD_NAME, "Conv2D layer expects 4D shapes");
		if (shapes.size() == 2 && shapes[0] != shapes[1])
			throw IllegalArgument(METHOD_NAME, "Conv2D layer expects both input shapes to be equal");

		m_input_shapes = shapes;
		if (m_padding == ConvPadding::VALID)
			m_config.padding.fill(0);
		else
		{

		}
	}
	Shape Conv2D::getOutputShape() const
	{
		return math::getConvolutionOutputShape(m_config, getInputShape(), getWeightShape());
	}
	Shape Conv2D::getWeightShape() const
	{
		return Shape( { m_output_filters, m_config.kernel[0], m_config.kernel[1], getInputShape().lastDim() });
	}
	Shape Conv2D::getBiasShape() const
	{
		if (m_use_bias)
			return Shape( { m_output_filters });
		else
			return Shape();
	}

	std::string Conv2D::name() const
	{
		return "Conv2D";
	}
	Json Conv2D::getConfig() const
	{
		Json result = Layer::getConfig();
		result["output_filters"] = m_output_filters;
		result["groups"] = m_config.groups;
		result["kernel"] = Json( { m_config.kernel[0], m_config.kernel[1] });
		result["stride"] = Json( { m_config.stride[0], m_config.stride[1] });
		result["dilation"] = Json( { m_config.dilation[0], m_config.dilation[1] });
		result["padding"] = static_cast<int>(m_padding);
		result["use_bias"] = m_use_bias;
		return result;
	}

	Conv2D* Conv2D::clone(const Json &config) const
	{
		std::unique_ptr<Conv2D> result = std::make_unique<Conv2D>(0, 0);
		result->m_output_filters = config["output_filters"];
		result->m_config.groups = config["groups"];
		result->m_config.kernel[0] = config["kernel"][0];
		result->m_config.kernel[1] = config["kernel"][1];
		result->m_config.stride[0] = config["stride"][0];
		result->m_config.stride[1] = config["stride"][1];
		result->m_config.dilation[0] = config["dilation"][0];
		result->m_config.dilation[1] = config["dilation"][1];
		result->m_padding = static_cast<ConvPadding>(config["padding"].getInt());
		result->m_use_bias = config["use_bias"];
		return result.release();
	}

	void Conv2D::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(same_device(context(), input[0], output));

//		switch (m_algorithm)
//		{
//			case ConvAlgorithm::DIRECT:
//				break;
//			case ConvAlgorithm::GEMM_IMPLICIT:
//				break;
//			case ConvAlgorithm::GEMM_EXPLICIT:
//			{
//				if (input.size() == 1)
//					math::convolution2D::explicitGemmForward(context(), input[0], output, getWeights().getParam(), getBias().getParam(), m_workspace,
//							m_nonlinearity);
//				else
//					math::convolution2D::explicitGemmForward(context(), input[0], output, getWeights().getParam(), getBias().getParam(), m_workspace,
//							m_nonlinearity, &(input[1]));
//				break;
//			}
//			case ConvAlgorithm::WINOGRAD_2:
//			{
//				Tensor weight_matrices = m_transformed_weights->view( { 16, m_output_filters, m_input_filters });
//				if (m_are_weights_transformed == false)
//				{
//					m_are_weights_transformed = true;
//					math::winograd2x2TransformWeight(context(), getWeights().getParam(), weight_matrices, false);
//				}
//
//				Tensor input_matrices = m_workspace.view( { 16, winograd_nb_of_tiles<2>(input[0].shape()), m_input_filters });
//				Tensor output_matrices = m_workspace.view( { 16, winograd_nb_of_tiles<2>(output.shape()), m_output_filters }, input_matrices.volume());
//				math::winograd2x2TransformInput(context(), input[0], input_matrices);
//				math::gemmBatched(context(), 'n', 't', output_matrices, input_matrices, weight_matrices);
//				if (input.size() == 1)
//					math::winograd2x2TransformOutput(context(), output, output_matrices, getBias().getParam(), nullptr, m_nonlinearity);
//				else
//					math::winograd2x2TransformOutput(context(), output, output_matrices, getBias().getParam(), &(input[1]), m_nonlinearity);
//				break;
//			}
//			case ConvAlgorithm::WINOGRAD_4:
//			{
//				Tensor weight_matrices = m_transformed_weights->view( { 36, m_output_filters, m_input_filters });
//				if (m_are_weights_transformed == false)
//				{
//					m_are_weights_transformed = true;
//					math::winograd4x4TransformWeight(context(), getWeights().getParam(), weight_matrices, false);
//				}
//
//				Tensor input_matrices = m_workspace.view( { 36, winograd_nb_of_tiles<4>(input[0].shape()), m_input_filters });
//				Tensor output_matrices = m_workspace.view( { 36, winograd_nb_of_tiles<4>(output.shape()), m_output_filters }, input_matrices.volume());
//				math::winograd4x4TransformInput(context(), input[0], input_matrices);
//				math::gemmBatched(context(), 'n', 't', output_matrices, input_matrices, weight_matrices);
//				if (input.size() == 1)
//					math::winograd4x4TransformOutput(context(), output, output_matrices, getBias().getParam(), nullptr, m_nonlinearity);
//				else
//					math::winograd4x4TransformOutput(context(), output, output_matrices, getBias().getParam(), &(input[1]), m_nonlinearity);
//				break;
//			}
//		}
	}
	void Conv2D::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next)
	{
		assert(same_device(context(), input[0], output, gradient_prev[0], gradient_next));

//		switch (m_algorithm)
//		{
//			case ConvAlgorithm::DIRECT:
//				break;
//			case ConvAlgorithm::GEMM_IMPLICIT:
//				break;
//			case ConvAlgorithm::GEMM_EXPLICIT:
//			{
//				math::convolution2D::explicitGemmBackward(context(), gradient_prev[0], gradient_next, output, getWeights().getParam(), m_workspace,
//						m_nonlinearity);
//				math::convolution2D::explicitGemmUpdate(context(), input[0], gradient_next, getWeights().getUpdate(), m_workspace);
//				break;
//			}
//			case ConvAlgorithm::WINOGRAD_2:
//			{
//				math::nonlinearityBackwardInPlace(context(), gradient_next, output, m_nonlinearity);
//				m_are_weights_transformed = false;
//				Tensor weight_matrices = m_transformed_weights->view( { 16, m_output_filters, m_input_filters });
//				math::winograd2x2TransformWeight(context(), getWeights().getParam(), weight_matrices, true);
//
//				Tensor gradient_next_matrices = m_workspace.view( { 16, winograd_nb_of_tiles<2>(output.shape()), m_output_filters });
//				Tensor gradient_prev_matrices = m_workspace.view( { 16, winograd_nb_of_tiles<2>(input[0].shape()), m_input_filters },
//						gradient_next_matrices.volume());
//
//				math::winograd2x2TransformInput(context(), gradient_next, gradient_next_matrices);
//				math::gemmBatched(context(), 'n', 'n', gradient_prev_matrices, gradient_next_matrices, weight_matrices);
//				math::winograd2x2TransformOutput(context(), gradient_prev[0], gradient_prev_matrices, Tensor( { }, dtype(), device()));
//
//				Tensor weight_update_matrices = m_transformed_weights->view( { 16, m_output_filters, m_input_filters });
//				math::winograd2x2TransformGradient(context(), gradient_next, gradient_next_matrices);
//				math::winograd2x2TransformInput(context(), input[0], gradient_prev_matrices);
//				math::gemmBatched(context(), 't', 'n', weight_update_matrices, gradient_next_matrices, gradient_prev_matrices);
//				math::winograd2x2TransformUpdate(context(), getWeights().getUpdate(), weight_update_matrices);
//				break;
//			}
//			case ConvAlgorithm::WINOGRAD_4:
//			{
//				math::nonlinearityBackwardInPlace(context(), gradient_next, output, m_nonlinearity);
//				m_are_weights_transformed = false;
//				Tensor weight_matrices = m_transformed_weights->view( { 36, m_output_filters, m_input_filters });
//				math::winograd4x4TransformWeight(context(), getWeights().getParam(), weight_matrices, true);
//
//				Tensor gradient_next_matrices = m_workspace.view( { 36, winograd_nb_of_tiles<4>(output.shape()), m_output_filters });
//				Tensor gradient_prev_matrices = m_workspace.view( { 36, winograd_nb_of_tiles<4>(input[0].shape()), m_input_filters },
//						gradient_next_matrices.volume());
//
//				math::winograd4x4TransformInput(context(), gradient_next, gradient_next_matrices);
//				math::gemmBatched(context(), 'n', 'n', gradient_prev_matrices, gradient_next_matrices, weight_matrices);
//				math::winograd4x4TransformOutput(context(), gradient_prev[0], gradient_prev_matrices, Tensor( { }, dtype(), device()));
//
//				Tensor weight_update_matrices = m_transformed_weights->view( { 36, m_output_filters, m_input_filters });
//				math::winograd4x4TransformGradient(context(), gradient_next, gradient_next_matrices);
//				math::winograd4x4TransformInput(context(), input[0], gradient_prev_matrices);
//				math::gemmBatched(context(), 't', 'n', weight_update_matrices, gradient_next_matrices, gradient_prev_matrices);
//				math::winograd4x4TransformUpdate(context(), getWeights().getUpdate(), weight_update_matrices);
//				break;
//			}
//		}
//		if (m_use_bias)
//			math::sumOverFirstDim(context(), getBias().getUpdate(), gradient_next, m_workspace, 1);
//		if (gradient_prev.size() == 2)
//			gradient_prev[1].copyFrom(context(), gradient_next);
	}
} /* namespace avocado */

