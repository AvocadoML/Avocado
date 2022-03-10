/*
 * placeholder_cpu.cpp
 *
 *  Created on: Nov 30, 2021
 *      Author: Maciej Kozarzewski
 */

#include <CpuBackend/cpu_backend.h>

namespace avocado
{
	namespace backend
	{
//		avStatus_t cpuGetDeviceProperty(avDeviceProperty_t propertyName, void *result)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuSetNumberOfThreads(int threads)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		int cpuGetNumberOfThreads()
//		{
//			return 0;
//		}
//
//		avStatus_t cpuCreateContextDescriptor(avContextDescriptor_t *result)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuDestroyContextDescriptor(avContextDescriptor_t desc)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avContextDescriptor_t cpuGetDefaultContext()
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuSynchronizeWithContext(avContextDescriptor_t context)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuIsContextReady(avContextDescriptor_t context, bool *result)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuCreateMemoryDescriptor(avMemoryDescriptor_t *result, avSize_t sizeInBytes)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuCreateMemoryView(avMemoryDescriptor_t *result, const avMemoryDescriptor_t desc, avSize_t sizeInBytes, avSize_t offsetInBytes)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuDestroyMemoryDescriptor(avMemoryDescriptor_t desc)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuSetMemory(avContextDescriptor_t context, avMemoryDescriptor_t dst, avSize_t dstOffset, avSize_t dstSize, const void *pattern,
//				avSize_t patternSize)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuCopyMemory(avContextDescriptor_t context, avMemoryDescriptor_t dst, avSize_t dstOffset, const avMemoryDescriptor_t src,
//				avSize_t srcOffset, avSize_t count)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		void* cpuGetMemoryPointer(avMemoryDescriptor_t mem)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuCreateTensorDescriptor(avTensorDescriptor_t *result)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuDestroyTensorDescriptor(avTensorDescriptor_t desc)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuSetTensorDescriptor(avTensorDescriptor_t desc, avDataType_t dtype, int nbDims, const int dimensions[])
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuGetTensorDescriptor(avTensorDescriptor_t desc, avDataType_t *dtype, int *nbDims, int dimensions[])
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuCreateConvolutionDescriptor(avConvolutionDescriptor_t *result)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuDestroyConvolutionDescriptor(avConvolutionDescriptor_t desc)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuSetConvolutionDescriptor(avConvolutionDescriptor_t desc, avConvolutionMode_t mode, int nbDims, const int padding[],
//				const int strides[], const int dilation[], int groups, const void *paddingValue)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuGetConvolutionDescriptor(avConvolutionDescriptor_t desc, avConvolutionMode_t *mode, int *nbDims, int padding[], int strides[],
//				int dilation[], int *groups, void *paddingValue)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuCreateOptimizerDescriptor(avOptimizerDescriptor_t *result)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuDestroyOptimizerDescriptor(avOptimizerDescriptor_t desc)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuSetOptimizerSGD(avOptimizerDescriptor_t desc, double learningRate, bool useMomentum, bool useNesterov, double beta1)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuGetOptimizerSGD(avOptimizerDescriptor_t desc, double *learningRate, bool *useMomentum, bool *useNesterov, double *beta1)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuSetOptimizerADAM(avOptimizerDescriptor_t desc, double learningRate, double beta1, double beta2)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuGetOptimizerADAM(avOptimizerDescriptor_t desc, double *learningRate, double *beta1, double *beta2)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuGetOptimizerType(avOptimizerDescriptor_t desc, avOptimizerType_t *type)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuChangeType(avContextDescriptor_t context, avMemoryDescriptor_t dst, avDataType_t dstType, const avMemoryDescriptor_t src,
//				avDataType_t srcType, avSize_t elements)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuConcatTensors(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
//				const avTensorDescriptor_t aDesc[], const avMemoryDescriptor_t aMem[], int nbTensors)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuSplitTensors(avContextDescriptor_t context, const avTensorDescriptor_t cDesc[], avMemoryDescriptor_t cMem[],
//				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, int nbTensors)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuTranspose(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
//				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const int newDimOrder[])
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuScaleTensor(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, const void *alpha)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuAddScalarToTensor(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
//				const void *scalar)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuBinaryOp(avContextDescriptor_t context, avBinaryOp_t operation, const void *alpha1, const avTensorDescriptor_t aDesc,
//				const avMemoryDescriptor_t aMem, const void *alpha2, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
//				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuUnaryOp(avContextDescriptor_t context, avUnaryOp_t operation, const void *alpha, const avTensorDescriptor_t aDesc,
//				const avMemoryDescriptor_t aMem, const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuReduceTensor(avContextDescriptor_t context, avReduceOp_t operation, const void *alpha, const avTensorDescriptor_t aDesc,
//				const avMemoryDescriptor_t aMem, const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuAddBias(avContextDescriptor_t context, const void *alpha3, const void *alpha1, const avTensorDescriptor_t aDesc,
//				const avMemoryDescriptor_t aMem, const void *alpha2, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
//				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, avActivationType_t activation)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuGemm(avContextDescriptor_t context, avGemmOperation_t aOp, avGemmOperation_t bOp, const void *alpha,
//				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
//				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuGemmBatched(avContextDescriptor_t context, avGemmOperation_t aOp, avGemmOperation_t bOp, const void *alpha,
//				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
//				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuActivationForward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
//				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
//				avMemoryDescriptor_t yMem)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuActivationBackward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
//				const avTensorDescriptor_t yDesc, const avMemoryDescriptor_t yMem, const avTensorDescriptor_t dyDesc,
//				const avMemoryDescriptor_t dyMem, const void *beta, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuSoftmaxForward(avContextDescriptor_t context, avSoftmaxMode_t mode, const void *alpha, const avTensorDescriptor_t xDesc,
//				const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuSoftmaxBackward(avContextDescriptor_t context, avSoftmaxMode_t mode, const void *alpha, const avTensorDescriptor_t yDesc,
//				const avMemoryDescriptor_t yMem, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem, const void *beta,
//				const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuAffineForward(avContextDescriptor_t context, avActivationType_t activation, const avTensorDescriptor_t wDesc,
//				const avMemoryDescriptor_t wMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha,
//				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
//				avMemoryDescriptor_t yMem)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuBatchNormInference(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
//				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
//				avMemoryDescriptor_t yMem, const avTensorDescriptor_t scaleBiasMeanVarDesc, const avMemoryDescriptor_t scaleMem,
//				const avMemoryDescriptor_t biasMem, const avMemoryDescriptor_t meanMem, const avMemoryDescriptor_t varianceMem, double epsilon)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuBatchNormForward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
//				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
//				avMemoryDescriptor_t yMem, const avTensorDescriptor_t scaleBiasMeanVarDesc, const avMemoryDescriptor_t scaleMem,
//				const avMemoryDescriptor_t biasMem, avMemoryDescriptor_t meanMem, avMemoryDescriptor_t varianceMem, double epsilon)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuBatchNormBackward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
//				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t yDesc, const avMemoryDescriptor_t yMem,
//				const void *beta, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem, const avTensorDescriptor_t dyDesc,
//				avMemoryDescriptor_t dyMem, const avTensorDescriptor_t scaleMeanVarDesc, const avMemoryDescriptor_t scaleMem,
//				const avMemoryDescriptor_t meanMem, const avMemoryDescriptor_t varianceMem, const void *alpha2, const void *beta2,
//				avMemoryDescriptor_t scaleUpdateMem, avMemoryDescriptor_t biasUpdateMem, double epsilon)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuDropoutForward(avContextDescriptor_t context, const avDropoutDescriptor_t config, const avTensorDescriptor_t xDesc,
//				const avMemoryDescriptor_t xMem, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem, avMemoryDescriptor_t states)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuDropoutBackward(avContextDescriptor_t context, const avDropoutDescriptor_t config, const avTensorDescriptor_t dyDesc,
//				const avMemoryDescriptor_t dyMem, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem, const avTensorDescriptor_t states)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuPoolingForward(avContextDescriptor_t context, const avPoolingDescriptor_t config, const void *alpha,
//				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
//				avMemoryDescriptor_t yMem)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuPoolingBackward(avContextDescriptor_t context, const avPoolingDescriptor_t config, const void *alpha,
//				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t dyDesc,
//				const avMemoryDescriptor_t dyMem, const void *beta, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuIm2Row(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t filterDesc,
//				const avTensorDescriptor_t srcDesc, const avMemoryDescriptor_t srcMem, const avTensorDescriptor_t rowDesc,
//				avMemoryDescriptor_t rowMem)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuGetConvolutionWorkspaceSize(const avConvolutionDescriptor_t config, const avTensorDescriptor_t xDesc,
//				const avTensorDescriptor_t wDesc, const avTensorDescriptor_t bDesc, avSize_t *result)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuConvolutionBiasActivationForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
//				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
//				const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2, const avTensorDescriptor_t zDesc,
//				const avMemoryDescriptor_t zMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
//				const avActivationType_t activation, avMemoryDescriptor_t workspace)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuConvolutionForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
//				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
//				const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuConvolutionBackward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
//				const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
//				const void *beta, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem, avMemoryDescriptor_t workspaceMem)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuConvolutionUpdate(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
//				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t dyDesc,
//				const avMemoryDescriptor_t dyMem, const void *beta, const avTensorDescriptor_t dwDesc, avMemoryDescriptor_t dwMem)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuMetricFunction(avContextDescriptor_t context, avMetricType_t metricType, const avTensorDescriptor_t outputDesc,
//				const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc, const avMemoryDescriptor_t targetMem, void *result)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuLossFunction(avContextDescriptor_t context, avLossType_t lossType, const avTensorDescriptor_t outputDesc,
//				const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc, const avMemoryDescriptor_t targetMem, void *result)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuLossGradient(avContextDescriptor_t context, avLossType_t lossType, const void *alpha, const avTensorDescriptor_t outputDesc,
//				const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc, const avMemoryDescriptor_t targetMem, const void *beta,
//				const avTensorDescriptor_t gradientDesc, avMemoryDescriptor_t gradientMem, bool isFused)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuGetOptimizerWorkspaceSize(avOptimizerDescriptor_t desc, const avTensorDescriptor_t wDesc, avSize_t *result)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuOptimizerLearn(avContextDescriptor_t context, const avOptimizerDescriptor_t config, const avTensorDescriptor_t wDesc,
//				avMemoryDescriptor_t wMem, const avTensorDescriptor_t dwDesc, const avTensorDescriptor_t dwMem, avMemoryDescriptor_t workspace)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t cpuRegularizerL2(avContextDescriptor_t context, const avTensorDescriptor_t gradientDesc, avMemoryDescriptor_t gradientMem,
//				const avTensorDescriptor_t weightDesc, const avMemoryDescriptor_t weightMem, const void *coefficient, const void *offset, void *loss)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
	} /* namespace backend */
} /* namespace backend */

