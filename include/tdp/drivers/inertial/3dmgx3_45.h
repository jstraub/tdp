
#pragma once
#include <string>
#include <iostream>
#include <asio.hpp>
#include <asio/serial_port.hpp>
#include <tdp/inertial/imu_interface.h>

namespace tdp {


class Imu3DMGX3_45 : public ImuInterface {
 public:

  Imu3DMGX3_45(const std::string& port) : io_(), 
    serial_(io_), port_(port), 
    timeout_(boost::posix_time::seconds(1.0)), 
    timer_(io_, timeout_),
    bytesTransferred_(0) {}
  virtual ~Imu3DMGX3_45() {}

  virtual void GrabNext(ImuObs& obs);

  virtual void Start();
  virtual void Stop();

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

 private:

  static constexpr uint8_t BYTE_SYNC1 = 0x75;
  static constexpr uint8_t BYTE_SYNC2 = 0x65;

  asio::io_service io_;
  asio::serial_port serial_;
  std::string port_;
  asio::deadline_timer timer_; // timer for timeout
  boost::posix_time::time_duration timeout_;
  asio::streambuf readData_;
  size_t bytesTransferred_;

  bool SendNoDataCmd(uint8_t cmd_set, uint8_t cmd);
  enum ReadResult { resultInProgress, resultSuccess, resultError, resultTimeoutExpired};
  enum ReadResult result;

  void AppendCRC(std::vector<uint8_t>& arr);
  bool CheckCRC(std::vector<uint8_t>& arr);
  bool CheckACK(std::vector<uint8_t>& arr, uint8_t cmd_set, uint8_t cmd);

  void readCompleted(const asio::error_code& error,
      const size_t bytesTransferred) {
    if(!error) {
      result=resultSuccess;
      this->bytesTransferred_ = bytesTransferred;
      return;
    }
    if(error.value()==125) return; //Linux outputs error 125
    result=resultError;
  }

  void timeoutExpired(const asio::error_code& error)
  {
    if(!error && result==resultInProgress) result=resultTimeoutExpired;
  }

  void Read(char *data, size_t size)
  {
    if(readData_.size()>0) //If there is some data from a previous read
    {
      std::basic_istream<char> is(&readData_);
      size_t toRead=std::min(readData_.size(),size);//How many bytes to read?
      is.read(data,toRead);
      data+=toRead;
      size-=toRead;
      if(size==0) return;//If read data was enough, just return
    }

    result=resultInProgress;
    bytesTransferred_=0;

    asio::async_read(serial_,asio::buffer(data,size),
        [=] (const asio::error_code& error,
          const size_t bytesTransferred) {
        if(!error) {
        result=resultSuccess;
        this->bytesTransferred_ = bytesTransferred;
        return;
        }
        if(error.value()==125) return; //Linux outputs error 125
        result=resultError;
        });
//(
//asio::bind(
//          &Imu3DMGX3_45::readCompleted,this,asio::placeholders::error,
//          asio::placeholders::bytes_transferred)
//);

    //For this code to work, there should always be a timeout, so the
    //request for no timeout is translated into a very long timeout
    if(timeout_!=boost::posix_time::seconds(0)) 
      timer_.expires_from_now(timeout_);
    else 
      timer_.expires_from_now(boost::posix_time::hours(100000));

    timer_.async_wait(
        [=](const asio::error_code& error)
        {
        if(!error && result==resultInProgress) result=resultTimeoutExpired;
        });
//std::bind(&Imu3DMGX3_45::timeoutExpired,this,
//          asio::placeholders::error));

    for(;;)
    {
      io_.run_one();
      switch(result)
      {
        case resultInProgress:
          continue;
        case resultSuccess:
          timer_.cancel();
          return;
        case resultTimeoutExpired:
          serial_.cancel();
          std::cerr << "timeout exception" << std::endl;
        case resultError:
          timer_.cancel();
          serial_.cancel();
          throw(asio::system_error(asio::error_code(),
                "Error while reading"));
          //if resultInProgress remain in the loop
      }
    }
  }

  std::vector<uint8_t> Read(size_t size)
  {
    //Allocate a vector with the desired size
    std::vector<uint8_t> result(size,'\0');
    Read((char*)&result[0],size);//Fill it with values
    return result;
  }

  void waitForMsg() {
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

  void Write(const std::vector<uint8_t>& data)
  {
    asio::write(serial_,asio::buffer(&data[0],data.size()));
  }

};

}
