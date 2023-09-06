#include <iostream>
#include <Image.h>
#include "pub.h"
#include <fastrtps/attributes/ParticipantAttributes.h>
#include <fastrtps/attributes/PublisherAttributes.h>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>
#include <fastdds/dds/publisher/qos/PublisherQos.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/publisher/qos/DataWriterQos.hpp>
#include <ImagePubSubTypes.h>

#include <thread>

using namespace eprosima::fastdds::dds;

Pub::Pub()
    : participant_(nullptr), publisher_(nullptr), topic_(nullptr), writer_(nullptr), stop_(false), type_(new sys::msg::ImagePubSubType()) {
}

bool Pub::init(
    bool use_env) {
  image_.head().seq(0);
  DomainParticipantQos pqos = PARTICIPANT_QOS_DEFAULT;
  pqos.name("Participant_pub");
  auto factory = DomainParticipantFactory::get_instance();

  if (use_env) {
    factory->load_profiles();
    factory->get_default_participant_qos(pqos);
  }

  participant_ = factory->create_participant(0, pqos);

  if (participant_ == nullptr) {
    return false;
  }

  // REGISTER THE TYPE
  type_.register_type(participant_);

  // CREATE THE PUBLISHER
  PublisherQos pubqos = PUBLISHER_QOS_DEFAULT;

  if (use_env) {
    participant_->get_default_publisher_qos(pubqos);
  }

  publisher_ = participant_->create_publisher(
      pubqos,
      nullptr);

  if (publisher_ == nullptr) {
    return false;
  }

  // CREATE THE TOPIC
  TopicQos tqos = TOPIC_QOS_DEFAULT;

  if (use_env) {
    participant_->get_default_topic_qos(tqos);
  }

  topic_ = participant_->create_topic(
      "HelloWorldTopic",
      type_.get_type_name(),
      tqos);

  if (topic_ == nullptr) {
    return false;
  }

  // CREATE THE WRITER
  DataWriterQos wqos = DATAWRITER_QOS_DEFAULT;

  if (use_env) {
    publisher_->get_default_datawriter_qos(wqos);
  }

  writer_ = publisher_->create_datawriter(
      topic_,
      wqos,
      &listener_);

  return writer_ != nullptr;
}

Pub::~Pub() {
  if (writer_ != nullptr) {
    publisher_->delete_datawriter(writer_);
  }
  if (publisher_ != nullptr) {
    participant_->delete_publisher(publisher_);
  }
  if (topic_ != nullptr) {
    participant_->delete_topic(topic_);
  }
  DomainParticipantFactory::get_instance()->delete_participant(participant_);
}

void Pub::PubListener::on_publication_matched(
    eprosima::fastdds::dds::DataWriter* /*writer*/,
    const eprosima::fastdds::dds::PublicationMatchedStatus& info) {
  if (info.current_count_change == 1) {
    matched_ = info.total_count;
    firstConnected_ = true;
    std::cout << "Publisher matched." << std::endl;
  } else if (info.current_count_change == -1) {
    matched_ = info.total_count;
    std::cout << "Publisher unmatched." << std::endl;
  } else {
    std::cout << info.current_count_change
              << " is not a valid value for PublicationMatchedStatus current count change" << std::endl;
  }
}

void Pub::runThread(
    uint32_t samples,
    uint32_t sleep) {
  if (samples == 0) {
    while (!stop_) {
      if (publish(false)) {
        std::cout << "Seq: " << image_.head().seq()
                  << " SENT" << std::endl;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep));
    }
  } else {
    for (uint32_t i = 0; i < samples; ++i) {
      if (!publish()) {
        --i;
      } else {
        std::cout << "Seq: " << image_.head().seq()
                  << " SENT" << std::endl;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep));
    }
  }
}

void Pub::run(
    uint32_t samples,
    uint32_t sleep) {
  stop_ = false;
  std::thread thread(&Pub::runThread, this, samples, sleep);
  if (samples == 0) {
    std::cout << "Publisher running. Please press enter to stop the Publisher at any time." << std::endl;
    std::cin.ignore();
    stop_ = true;
  } else {
    std::cout << "Publisher running " << samples << " samples." << std::endl;
  }
  thread.join();
}

bool Pub::publish(
    bool waitForListener) {
  if (listener_.firstConnected_ || !waitForListener || listener_.matched_ > 0) {
    image_.head().seq(image_.head().seq() + 1);
    writer_->write(&image_);
    return true;
  }
  return false;
}

int main() {
  Pub pub;

  pub.init(false);
  pub.run(1, 2);

  return 0;
}
