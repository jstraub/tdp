/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#pragma once
#include <string>
#include <thread>
#include <iostream>
#include <asio.hpp>
#include <asio/serial_port.hpp>
#include <tdp/inertial/imu_interface.h>
#include <tdp/utils/threadedValue.hpp>
#include <tdp/data/circular_buffer.h>
#include <pangolin/utils/picojson.h>

namespace tdp {

// inspired by
// https://github.com/ZdenekM/microstrain_3dm_gx3_45/blob/master/include/microstrain_3dm_gx3_45/driver.h

class Imu3DMGX3_45 : public ImuInterface {
 public:

  Imu3DMGX3_45(const std::string& port, int rate) : io_(), 
    serial_(io_), port_(port), 
    timer_(io_, timeout_),
    timeout_(boost::posix_time::seconds(10.0)), 
    bytesTransferred_(0) ,
    rate_(rate),
    circBuf_(10000)
  {}
  virtual ~Imu3DMGX3_45() {Stop();}

  virtual bool GrabNext(ImuObs& obs);
  virtual bool GrabNewest(ImuObs& obs);

  virtual void Start();
  virtual void Stop();

  virtual pangolin::json::value GetProperties() {
    return properties_;
  }

  enum cmd_set {
    CMD_SET_BASIC = 0x01,
    CMD_SET_3DM = 0x0C,
    CMD_SET_NAVFILTER = 0x0D,
    CMD_SET_SYSTEM = 0x7F
  };
  enum cmd_set_basic {
    CMD_BASIC_PING = 0x01,
    CMD_BASIC_SET_TO_IDLE = 0x02,
    CMD_BASIC_GET_DEV_INFO = 0x03,
    CMD_BASIC_GET_DEV_DESC_SETS = 0x04,
    CMD_BASIC_DEV_BUILTIN_TEST = 0x05,
    CMD_BASIC_RESUME = 0x06,
    CMD_BASIC_RESET = 0x7E
  };
  enum cmd_set_3dm {
    CMD_3DM_POLL_AHRS = 0x01,
    CMD_3DM_POLL_GPS = 0x02,
    CMD_3DM_POLL_NAV = 0x03,
    CMD_3DM_DEV_STATUS = 0x64,
    CMD_3DM_STREAM_STATE = 0x11,
    CMD_3DM_AHRS_MSG_FORMAT = 0x08,
    CMD_3DM_GPS_MSG_FORMAT = 0x09,
    CMD_3DM_NAV_MSG_FORMAT = 0x0A
  };
  enum comm_modes {
    COMM_MODE_MIP = 0x01,
    COMM_MODE_AHRS = 0x02,
    COMM_MODE_GPS = 0x03
  };
  enum others {
    MODEL_ID = 0x1854,
    DATA_AHRS = 0x80,
    DATA_GPS = 0x81
  };
  enum functions {
    FUN_USE_NEW = 0x01,
    FUN_READ_CURRENT = 0x02,
    FUN_SAVE_CURR_AS_STARTUP = 0x03,
    FUN_LOAD_SAVE_STARTUP = 0x04,
    FUN_RESET_TO_FACTORY_DEF = 0x05
  };
  enum streams {
    STREAM_AHRS = 0x01,
    STREAM_GPS  = 0x02,
    STREAM_NAV  = 0x03
  };
  enum ReadResult { 
    resultInProgress, 
    resultSuccess, 
    resultError,
    resultTimeoutExpired
  };

 private:

  static const uint8_t BYTE_SYNC1 = 0x75;
  static const uint8_t BYTE_SYNC2 = 0x65;

  asio::io_service io_;
  asio::serial_port serial_;
  std::string port_;
  asio::deadline_timer timer_; // timer for timeout
  boost::posix_time::time_duration timeout_;
  asio::streambuf readData_;
  size_t bytesTransferred_;

  int rate_;
  std::thread receivingThread_;
  tdp::ThreadedValue<bool> receive_;
 
  std::mutex circBufMutex_;
  tdp::ManagedHostCircularBuffer<ImuObs> circBuf_;

  pangolin::json::value properties_;

  bool SelfTest();
  bool SetAHRSMsgFormat();
  bool SetStream(uint8_t stream, bool state);
  bool Resume() {
    return SendNoDataCmd(CMD_SET_BASIC, CMD_BASIC_RESUME);
  }
  bool SetToIdle() {
    return SendNoDataCmd(CMD_SET_BASIC, CMD_BASIC_SET_TO_IDLE);
  }
  bool Reset() {
    return SendNoDataCmd(CMD_SET_BASIC, CMD_BASIC_RESET);
  }

  bool PollAHRS(ImuObs& imuObs);
  bool ReceiveAHRS(ImuObs& imuObs);

  bool SendNoDataCmd(uint8_t cmd_set, uint8_t cmd);
  enum ReadResult result;

  void AppendCRC(std::vector<uint8_t>& arr);
  bool CheckCRC(std::vector<uint8_t>& arr);
  bool CheckACK(std::vector<uint8_t>& arr, uint8_t cmd_set, uint8_t cmd);

  void Read(char *data, size_t size);

  std::vector<uint8_t> Read(size_t size)
  {
    //Allocate a vector with the desired size
    std::vector<uint8_t> result(size,'\0');
    Read((char*)&result[0],size);//Fill it with values
    return result;
  }

  void WaitForMsg() {
    std::vector<uint8_t> recv; // TODO just for testing!!!!! rewrite it
    char prev = ' ';
    do {
      if (recv.size() > 0) 
        prev = recv[0];
      else 
        prev = ' ';
      recv.clear();
      recv = Read(1);
    } while (!(prev=='u' && recv[0]=='e'));
  }

  void Write(const std::vector<uint8_t>& data) {
    asio::write(serial_,asio::buffer(&data[0],data.size()));
  }

  float ExtractFloat(uint8_t* addr) {
    float tmp;
    *((uint8_t*)(&tmp) + 3) = *(addr);
    *((uint8_t*)(&tmp) + 2) = *(addr+1);
    *((uint8_t*)(&tmp) + 1) = *(addr+2);
    *((uint8_t*)(&tmp)) = *(addr+3);
    return tmp;
  }

  uint32_t ExtractUint32(uint8_t* addr) {
    uint32_t tmp;
    *((uint8_t*)(&tmp) + 3) = *(addr);
    *((uint8_t*)(&tmp) + 2) = *(addr+1);
    *((uint8_t*)(&tmp) + 1) = *(addr+2);
    *((uint8_t*)(&tmp)) = *(addr+3);
    return tmp;
  }

};

}
