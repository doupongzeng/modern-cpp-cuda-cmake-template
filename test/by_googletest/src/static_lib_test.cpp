#include <gtest/gtest.h>
#include <static_lib.h>

TEST(MyStaticLib, StandardOutput)
{
    testing::internal::CaptureStdout();
    static_lib_p();
    EXPECT_EQ("static_lib_p\n", testing::internal::GetCapturedStdout());
}
