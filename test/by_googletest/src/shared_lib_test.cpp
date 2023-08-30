#include <gtest/gtest.h>
#include <shared_lib.h>

TEST(MySharedLib, StandardOutput)
{
    testing::internal::CaptureStdout();
    shared_lib_p();
    EXPECT_EQ("shared_lib_p\n", testing::internal::GetCapturedStdout());
}
