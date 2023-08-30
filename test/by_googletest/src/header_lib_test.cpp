#include <gtest/gtest.h>
#include <header_lib.h>

TEST(MyHeaderLib, StandardOutput)
{
    testing::internal::CaptureStdout();
    header_p();
    EXPECT_EQ("header_lib\n", testing::internal::GetCapturedStdout());
}
