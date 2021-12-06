/*
 * backend_defs.h
 *
 *  Created on: Jul 29, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_BACKEND_BACKEND_DEFS_H_
#define AVOCADO_BACKEND_BACKEND_DEFS_H_

namespace avocado
{
	namespace backend
	{
#ifdef _WIN32
#  ifdef BUILDING_DLL
#    define DLL_PUBLIC __declspec(dllexport)
#  else
#    define DLL_PUBLIC __declspec(dllimport)
#  endif
#else
#  define DLL_PUBLIC
#endif

#ifdef __cplusplus
		extern "C"
		{
#endif

		typedef long long avSize_t;

		typedef enum
		{
			AVOCADO_DEVICE_CPU,
			AVOCADO_DEVICE_CUDA,
			AVOCADO_DEVICE_OPENCL
		} avDeviceType_t;
		typedef int avDeviceIndex_t;

		/**
		 *  Enumeration type used for function status returns, which can be one of the following values.
		 */
		typedef enum
		{
			AVOCADO_STATUS_SUCCESS, /**< The operation was completed successfully. */
			AVOCADO_STATUS_ALLOC_FAILED, /**< Resource allocation failed inside the library. */
			AVOCADO_STATUS_FREE_FAILED, /**< Resource deallocation failed inside the library. This is an irrecoverable error. */
			AVOCADO_STATUS_BAD_PARAM, /**< An incorrect value or parameter was passed to the function. */
			AVOCADO_STATUS_ARCH_MISMATCH, /**< The function requires a feature that is not supported on a device. */
			AVOCADO_STATUS_INTERNAL_ERROR, /**< An internal Avocado operation failed. */
			AVOCADO_STATUS_NOT_SUPPORTED, /**< The functionality requested is not presently supported. */
			AVOCADO_STATUS_UNSUPPORTED_DATATYPE /**< The data type is not presently supported. */
		} avStatus_t;

		/**
		 * Enumeration type indicating the data type tensors and scalars use.
		 */
		typedef enum
		{
			AVOCADO_DTYPE_UNKNOWN, /**< */
			AVOCADO_DTYPE_UINT8, /**< The data is an 8-bit unsigned integer. */
			AVOCADO_DTYPE_INT8, /**< The data is an 8-bit signed integer. */
			AVOCADO_DTYPE_INT16, /**< The data is an 16-bit signed integer. */
			AVOCADO_DTYPE_INT32, /**< The data is an 32-bit signed integer. */
			AVOCADO_DTYPE_INT64, /**< The data is an 64-bit signed integer. */
			AVOCADO_DTYPE_FLOAT16, /**< The data is a 16-bit floating-point. */
			AVOCADO_DTYPE_BFLOAT16, /**< The data is a 16-bit quantity, with 7 mantissa bits, 8 exponent bits, and 1 sign bit. */
			AVOCADO_DTYPE_FLOAT32, /**< The data is a 32-bit single-precision floating-point (float). */
			AVOCADO_DTYPE_FLOAT64, /**< The data is a 64-bit double-precision floating-point (double). */
			AVOCADO_DTYPE_COMPLEX32, /**< The data is a complex number 32-bit single-precision floating-point. */
			AVOCADO_DTYPE_COMPLEX64 /**< The data is a complex number 64-bit double-precision floating-point. */
		} avDataType_t;

		/**
		 * Enumeration type used to specify which activation function will be used.
		 */
		typedef enum
		{
			AVOCADO_ACTIVATION_LINEAR, /**< Selects identity function. */
			AVOCADO_ACTIVATION_SIGMOID, /**< Selects the sigmoid function. */
			AVOCADO_ACTIVATION_TANH, /**< Selects the hyperbolic tangent function. */
			AVOCADO_ACTIVATION_RELU, /**< Selects the clipped rectified linear function. */
			AVOCADO_ACTIVATION_SELU, /**< Selects the scaled exponential linear function. */
			AVOCADO_ACTIVATION_ELU, /**< Selects the exponential linear function. */
			AVOCADO_ACTIVATION_EXPONENTIAL, /**< Selects the exponential function. */
			AVOCADO_ACTIVATION_SOFTPLUS, /**< Selects the softplus function. */
			AVOCADO_ACTIVATION_SOFTSIGN, /**< Selects the softsign function. */
			AVOCADO_ACTIVATION_SOFTMAX /**< Selects the softmax function. */
		} avActivationType_t;

		/**
		 * Enumeration type used to indicate the operation to be used by the ReduceTensor() routine.
		 */
		typedef enum
		{
			AVOCADO_REDUCE_TENSOR_ADD, /**< The operation to be performed is addition. */
			AVOCADO_REDUCE_TENSOR_MUL, /**< The operation to be performed is multiplication. */
			AVOCADO_REDUCE_TENSOR_MIN, /**< The operation to be performed is a minimum comparison. */
			AVOCADO_REDUCE_TENSOR_MAX, /**< The operation to be performed is a maximum comparison. */
			AVOCADO_REDUCE_TENSOR_AMAX, /**< The operation to be performed is a maximum comparison of absolute values. */
			AVOCADO_REDUCE_TENSOR_AVG, /**< The operation to be performed is averaging. */
			AVOCADO_REDUCE_TENSOR_NORM1, /**< The operation to be performed is addition of absolute values. */
			AVOCADO_REDUCE_TENSOR_NORM2, /**< The operation to be performed is a square root of the sum of squares. */
			AVOCADO_REDUCE_TENSOR_MUL_NO_ZEROS /**< The operation to be performed is multiplication, not including elements of value zero. */
		} avReduceTensorOp_t;

//			CUDNN_POINTWISE_ABS
//			In this mode, a pointwise absolute value of the input tensor is computed.
//			CUDNN_POINTWISE_CEIL
//			In this mode, a pointwise ceiling of the input tensor is computed.
//			CUDNN_POINTWISE_COS
//			In this mode, a pointwise trigonometric cosine of the input tensor is computed.
//			CUDNN_POINTWISE_EXP
//			In this mode, a pointwise exponential of the input tensor is computed.
//			CUDNN_POINTWISE_FLOOR
//			In this mode, a pointwise floor of the input tensor is computed.
//			CUDNN_POINTWISE_LOG
//			In this mode, a pointwise natural logarithm of the input tensor is computed.
//			CUDNN_POINTWISE_NEG
//			In this mode, a pointwise numerical negative of the input tensor is computed.
//			CUDNN_POINTWISE_RSQRT
//			In this mode, a pointwise reciprocal of the square root of the input tensor is computed.
//			CUDNN_POINTWISE_SIN
//			In this mode, a pointwise trigonometric sine of the input tensor is computed.
//			CUDNN_POINTWISE_SQRT
//			In this mode, a pointwise square root of the input tensor is computed.
//			CUDNN_POINTWISE_TAN
//			In this mode, a pointwise trigonometric tangent of the input tensor is computed.
//			CUDNN_POINTWISE_LOGICAL_NOT
//			In this mode, a pointwise truth value of input tensor's logical NOT is computed.

//			CUDNN_POINTWISE_ADD
//			In this mode, a pointwise addition between two tensors is computed.
//			CUDNN_POINTWISE_ADD_SQUARE
//			In this mode, a pointwise addition between the first tensor and the square of the second tensor is computed.
//			CUDNN_POINTWISE_DIV
//			In this mode, a pointwise true division of the first tensor by second tensor is computed.
//			CUDNN_POINTWISE_MAX
//			In this mode, a pointwise maximum is taken between two tensors.
//			CUDNN_POINTWISE_MIN
//			In this mode, a pointwise minimum is taken between two tensors.
//			CUDNN_POINTWISE_MOD
//			In this mode, a pointwise floating-point remainder of the first tensor's division by the second tensor is computed.
//			CUDNN_POINTWISE_MUL
//			In this mode, a pointwise multiplication between two tensors is computed.
//			CUDNN_POINTWISE_POW
//			In this mode, a pointwise value from the first tensor to the power of the second tensor is computed.
//			CUDNN_POINTWISE_SUB
//			In this mode, a pointwise subtraction between two tensors is computed.
//			CUDNN_POINTWISE_CMP_EQ
//			In this mode, a pointwise truth value of the first tensor equal to the second tensor is computed.
//			CUDNN_POINTWISE_CMP_NEQ
//			In this mode, a pointwise truth value of the first tensor not equal to the second tensor is computed.
//			CUDNN_POINTWISE_CMP_GT
//			In this mode, a pointwise truth value of the first tensor greater than the second tensor is computed.
//			CUDNN_POINTWISE_CMP_GE
//			In this mode, a pointwise truth value of the first tensor greater than equal to the second tensor is computed.
//			CUDNN_POINTWISE_CMP_LT
//			In this mode, a pointwise truth value of the first tensor less than the second tensor is computed.
//			CUDNN_POINTWISE_CMP_LE
//			In this mode, a pointwise truth value of the first tensor less than equal to the second tensor is computed.
//			CUDNN_POINTWISE_LOGICAL_AND
//			In this mode, a pointwise truth value of the first tensor logical AND second tensor is computed.
//			CUDNN_POINTWISE_LOGICAL_OR
//			In this mode, a pointwise truth value of the first tensor logical OR second tensor is computed.

//			CUDNN_POINTWISE_RELU_FWD
//			In this mode, a pointwise rectified linear activation function of the input tensor is computed.
//			CUDNN_POINTWISE_TANH_FWD
//			In this mode, a pointwise tanh activation function of the input tensor is computed.
//			CUDNN_POINTWISE_SIGMOID_FWD
//			In this mode, a pointwise sigmoid activation function of the input tensor is computed.
//			CUDNN_POINTWISE_ELU_FWD
//			In this mode, a pointwise Exponential Linear Unit activation function of the input tensor is computed.
//			CUDNN_POINTWISE_GELU_FWD
//			In this mode, a pointwise Gaussian Error Linear Unit activation function of the input tensor is computed.
//			CUDNN_POINTWISE_SOFTPLUS_FWD
//			In this mode, a pointwise softplus activation function of the input tensor is computed.
//			CUDNN_POINTWISE_SWISH_FWD
//			In this mode, a pointwise swish activation function of the input tensor is computed.
//			CUDNN_POINTWISE_RELU_BWD
//			In this mode, a pointwise first derivative of rectified linear activation of the input tensor is computed.
//			CUDNN_POINTWISE_TANH_BWD
//			In this mode, a pointwise first derivative of tanh activation of the input tensor is computed.
//			CUDNN_POINTWISE_SIGMOID_BWD
//			In this mode, a pointwise first derivative of sigmoid activation of the input tensor is computed.
//			CUDNN_POINTWISE_ELU_BWD
//			In this mode, a pointwise first derivative of Exponential Linear Unit activation of the input tensor is computed.
//			CUDNN_POINTWISE_GELU_BWD
//			In this mode, a pointwise first derivative of Gaussian Error Linear Unit activation of the input tensor is computed.
//			CUDNN_POINTWISE_SOFTPLUS_BWD
//			In this mode, a pointwise first derivative of softplus activation of the input tensor is computed.
//			CUDNN_POINTWISE_SWISH_BWD
//			In this mode, a pointwise first derivative of swish activation of the input tensor is computed.

		/**
		 * Enumeration type used to indicate the operation to be used by the OpTensor() routine.
		 */
		typedef enum
		{
			AVOCADO_OP_TENSOR_ADD, /**< The operation to be performed is addition. */
			AVOCADO_OP_TENSOR_SUB, /**< The operation to be performed is subtraction. */
			AVOCADO_OP_TENSOR_MUL, /**< The operation to be performed is multiplication. */
			AVOCADO_OP_TENSOR_DIV, /**< The operation to be performed is division. */
			AVOCADO_OP_TENSOR_MIN, /**< The operation to be performed is a minimum comparison. */
			AVOCADO_OP_TENSOR_MAX /**< The operation to be performed is a maximum comparison. */
		} avOpTensorOp_t;

		/**
		 * Enumeration type used to indicate the operation to be used by the OpSingleTensor() routine.
		 */
		typedef enum
		{
			AVOCADO_OP_SINGLE_TENSOR_ABS, /**< The operation to be performed is absolute value. */
			AVOCADO_OP_SINGLE_TENSOR_SQUARE, /**< The operation to be performed is squaring. */
			AVOCADO_OP_SINGLE_TENSOR_SQRT, /**< The operation to be performed is square root. */
			AVOCADO_OP_SINGLE_TENSOR_NOT /**< The operation to be performed is negation. */
		} avOpSingleTensorOp_t;

		/**
		 * Enumeration type used to select the pooling method in PoolingForward() and PoolingBackward().
		 */
		typedef enum
		{
			AVOCADO_POOLING_MAX, /**< The maximum value inside the pooling window is used. */
			AVOCADO_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, /**< Values inside the pooling window are averaged including values from the padding region. */
			AVOCADO_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING /**< Values inside the pooling window are averaged excluding values from the padding region. */
		} avPoolingMode_t;

		/**
		 * Enumeration type indicating over which data the softmax function is calculated.
		 */
		typedef enum
		{
			AVOCADO_SOFTMAX_MODE_INSTANCE, /**< The softmax operation is computed per image (N) across the dimensions H,W,C. */
			AVOCADO_SOFTMAX_MODE_CHANNEL /**< The softmax operation is computed per image (N) and spatial location (H,W) across dimension C. */
		} avSoftmaxMode_t;

		typedef enum
		{
			AVOCADO_GEMM_OPERATION_N, /**< No operation is performed. */
			AVOCADO_GEMM_OPERATION_T, /**< The matrix is transposed. */
			AVOCADO_GEMM_OPERATION_C /**<  */
		} avGemmOperation_t;

		typedef enum
		{
			AVOCADO_ACCURACY_METRIC /**<  */
		} avMetricType_t;

		typedef enum
		{
			AVOCADO_MEAN_SQUARE_LOSS, /**<  */
			AVOCADO_CROSS_ENTROPY_LOSS, /**<  */
			AVOCADO_KL_DIVERGENCE_LOSS /**<  */
		} avLossType_t;

		typedef enum
		{
			AVOCADO_OPTIMIZER_SGD, /**< */
			AVOCADO_OPTIMIZER_ADAM /**< */
		} avOptimizerType_t;

		typedef enum
		{
			AVOCADO_CONV_ALGORITHM_AUTO, /**<  */
			AVOCADO_CONV_ALGORITHM_EXPLICIT_GEMM, /**<  */
			AVOCADO_CONV_ALGORITHM_IMPLICIT_GEMM, /**<  */
			AVOCADO_CONV_ALGORITHM_WINOGRAD, /**<  */
		} avConvAlgorithm_t;

		typedef int avContextDescriptor_t;
		typedef int avMemoryDescriptor_t;
		typedef int avTensorDescriptor_t;
		typedef int avConvolutionDescriptor_t;
		typedef int avPoolingDescriptor_t;

		/* Opaque structure holding context */
		struct ContextDescriptor;
		typedef struct ContextDescriptor *avContext_t;

		DLL_PUBLIC struct ShapeDescriptor
		{
				int dim[8];
				int length;
		};
		typedef struct ShapeDescriptor *avShape_t;

		DLL_PUBLIC struct ScalarDescriptor
		{
				char data[16];
				avDataType_t dtype;
		};
		typedef struct ScalarDescriptor *avScalar_t;

		DLL_PUBLIC struct TensorDescriptor
		{
				ShapeDescriptor shape;
				avDataType_t dtype;
				void *data;
		};
		typedef struct TensorDescriptor *avTensor_t;

		DLL_PUBLIC struct ConvolutionDescriptor
		{
				avConvAlgorithm_t algorithm;
				avActivationType_t activation;
				ShapeDescriptor filter;
				ShapeDescriptor padding;
				ShapeDescriptor stride;
				ShapeDescriptor dilation;
				ScalarDescriptor padding_value;
				int groups;
				bool invert_filter;
		};
		typedef struct ConvolutionDescriptor *avConvolution_t;

		DLL_PUBLIC struct PoolingDescriptor
		{
				avPoolingMode_t mode;
				ShapeDescriptor filter;
				ShapeDescriptor padding;
				ShapeDescriptor stride;
		};
		typedef struct PoolingDescriptor *avPooling_t;

		DLL_PUBLIC struct OptimizerDescriptor
		{
				avOptimizerType_t type;
				double learning_rate;
				double coef[4];
				bool flags[4];
		};
		typedef struct OptimizerDescriptor *avOptimizer_t;

		DLL_PUBLIC struct DropoutDescriptor
		{
				double propability;
		};
		typedef struct DropoutDescriptor *avDropout_t;

#ifdef __cplusplus
		}
#endif
	} /* namespace backend */
} /* namespace avocado */

#endif /* AVOCADO_BACKEND_BACKEND_DEFS_H_ */
