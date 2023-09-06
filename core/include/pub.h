#pragma once

#include "Image.h"

#include <fastdds/dds/publisher/DataWriterListener.hpp>
#include <fastdds/dds/topic/TypeSupport.hpp>
#include <fastdds/dds/domain/DomainParticipant.hpp>

class Pub 
{
public:

    Pub();

    Pub(const Pub & pub) = delete;
    Pub& operator =(const Pub &pub) = delete;

    Pub(Pub&& pub) noexcept = delete;
    Pub& operator=(Pub&& pub) noexcept = delete;

    virtual ~Pub();

    //!Initialize
    bool init(
            bool use_env);

    //!Publish a sample
    bool publish(
            bool waitForListener = true);

    //!Run for number samples
    void run(
            uint32_t samples,
            uint32_t sleep);

private:

    sys::msg::Image image_;

    eprosima::fastdds::dds::DomainParticipant* participant_;

    eprosima::fastdds::dds::Publisher* publisher_;

    eprosima::fastdds::dds::Topic* topic_;

    eprosima::fastdds::dds::DataWriter* writer_;

    bool stop_;

    class PubListener : public eprosima::fastdds::dds::DataWriterListener
    {
    public:

        PubListener()
            : matched_(0)
            , firstConnected_(false)
        {
        }
        PubListener(const PubListener& ) = delete;
        PubListener& operator=(const PubListener& ) = delete;

        PubListener(PubListener&& ) noexcept = delete;
        PubListener& operator=(PubListener&& ) noexcept = delete;

        ~PubListener() override = default;

        void on_publication_matched(
                eprosima::fastdds::dds::DataWriter* writer,
                const eprosima::fastdds::dds::PublicationMatchedStatus& info) override;

        int matched_;

        bool firstConnected_;
    }
    listener_;

    void runThread(
            uint32_t samples,
            uint32_t sleep);

    eprosima::fastdds::dds::TypeSupport type_;
};

