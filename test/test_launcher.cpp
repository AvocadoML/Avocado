/*
 * test_launcher.cpp
 *
 *  Created on: Jan 19, 2022
 *      Author: Maciej Kozarzewski
 */

#include <gtest/gtest.h>

int main(int argc, char *argv[])
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}


