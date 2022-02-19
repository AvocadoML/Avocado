//============================================================================
// Name        : Avocado.cpp
// Author      : Maciej Kozarzewski
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <Avocado/core/Context.hpp>
#include <Avocado/core/Tensor.hpp>
#include <Avocado/core/Scalar.hpp>
#include <Avocado/core/DataType.hpp>
#include <Avocado/math/tensor_operations.hpp>
#include <Avocado/utils/testing_helpers.hpp>

using namespace avocado;
using namespace avocado::backend;

int main()
{
	std::cout << Device::hardwareInfo() << std::endl;

	Context context(Device::cuda(0));
	Tensor t1( { 100 }, DataType::FLOAT16, Device::cuda(0));
	Tensor t2( { 100 }, DataType::FLOAT32, Device::cuda(0));
	initForTest(t1, 0.0f);
	initForTest(t2, 0.0f);

	Tensor t3(t1.shape(), DataType::FLOAT32, t1.device());
	math::changeType(context, t3, t1);

	std::cout << diffForTest(t2, t3) << '\n';

//	Context context(Device::cpu());
//	context.synchronize();
//
//	Tensor t1(Shape( { 10, 10, 10 }), DataType::FLOAT32, Device::cpu());
//	t1.setall(1.1f);
//	t1.set(2.5f, { 1, 2, 3 });
//
//	Tensor t2(Shape( { 10, 10, 10 }), DataType::INT32, Device::cpu());
//
//	math::changeType(context, t2, t1);
//	context.synchronize();
//
//	std::cout << t1.get<double>( { 1, 2, 3 }) << ' ' << t1.get<float>( { 1, 2, 2 }) << '\n';
//	std::cout << t2.get<int>( { 1, 2, 3 }) << ' ' << t2.get<int>( { 1, 2, 2 }) << '\n';

	std::cout << "!!!Hello World!!!" << std::endl; // prints !!!Hello World!!!
	return 0;
}
