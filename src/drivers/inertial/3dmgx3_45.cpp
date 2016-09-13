/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <tdp/drivers/inertial/3dmgx3_45.h>

namespace tdp {

//Imu3DMGX3_45::Imu3DMGX3_45() {
//}

//Imu3DMGX3_45::~Imu3DMGX3_45() {
//}

bool Imu3DMGX3_45::GrabNext(ImuObs& obs) {

  std::lock_guard<std::mutex> lock(circBufMutex_);
  if (circBuf_.SizeToRead() > 0) {
    obs = circBuf_.GetCircular(0);
    circBuf_.MarkRead(1);
    return true;
  }
  return false;
}

bool Imu3DMGX3_45::GrabNewest(ImuObs& obs) {
  std::lock_guard<std::mutex> lock(circBufMutex_);
  obs = circBuf_.GetCircular(-1);
  return true;
}

void Imu3DMGX3_45::Start() {
  if (serial_.is_open()) {
    Stop();
  }
  asio::serial_port_base::parity PARITY(asio::serial_port_base::parity::none);
  asio::serial_port_base::flow_control FLOW_CTRL(asio::serial_port_base::flow_control::none);
  asio::serial_port_base::stop_bits STOP(asio::serial_port_base::stop_bits::one);
  asio::serial_port_base::character_size CHAR_SIZE(8);

  serial_.open("/dev/ttyACM0");
  serial_.set_option(asio::serial_port_base::baud_rate(1000000));
  serial_.set_option(PARITY);
  serial_.set_option(CHAR_SIZE);
  serial_.set_option(FLOW_CTRL);
  serial_.set_option(STOP);

  //while(!Reset()) {}
  //std::cout << "reset device" << std::endl;

  if (SendNoDataCmd(CMD_SET_BASIC, CMD_BASIC_PING)) {
    std::cout << "ping successfull" << std::endl;
  }
  while(!SetToIdle()) {}
  std::cout << "set to idle" << std::endl;
  if (SetStream(STREAM_NAV,false)) {
    std::cout << "NAV stream stopped" << std::endl;
  }
  if (SetStream(STREAM_GPS,false)) {
    std::cout << "GPS stream stopped" << std::endl;
  }
  if (SetStream(STREAM_AHRS,false)) {
    std::cout << "AHRS stream stopped" << std::endl;
  }
  if (SelfTest()) {
    std::cout << "self test passed" << std::endl;
  } else {
    std::cout << "self test NOT passed! " << std::endl;
  }
  if (SetAHRSMsgFormat()) {
    std::cout << "AHRS format set successfully" << std::endl;
  }
  if (SetStream(STREAM_AHRS,true)) {
    std::cout << "AHRS stream started" << std::endl;
  }
  if (Resume()) {
    std::cout << "resuming started" << std::endl;
  }

  receive_.Set(true);
  receivingThread_ = std::thread([=](){
        ImuObs imuObs;
        std::cout << "started IMU receiver thread" << std::endl;
        while (receive_.Get()) {
          WaitForMsg();
          ReceiveAHRS(imuObs);
          {
            std::lock_guard<std::mutex> lock(circBufMutex_);
            circBuf_.Insert(imuObs);
          }
        }
      });
}

void Imu3DMGX3_45::Stop() {
  if (!serial_.is_open())
    return;
  receive_.Set(false);
  receivingThread_.join();
  if (SetToIdle()) {
    std::cout << "set to idle" << std::endl;
  }
  serial_.close();
}

bool Imu3DMGX3_45::SelfTest() {
  boost::posix_time::time_duration timeout_orig = timeout_;
	timeout_ = boost::posix_time::seconds(6);

  std::vector<uint8_t> data;
  uint8_t BYTE_SYNC1 = 0x75;
  uint8_t BYTE_SYNC2 = 0x65;

	data.push_back(BYTE_SYNC1);
	data.push_back(BYTE_SYNC2);
	data.push_back(CMD_SET_BASIC);
	data.push_back(0x02);
	data.push_back(0x02);
	data.push_back(CMD_BASIC_DEV_BUILTIN_TEST);

	AppendCRC(data);
	Write(data);
	WaitForMsg();

  std::vector<uint8_t> recv;
	size_t n = 14;

	recv = Read(n);

	if (!CheckCRC(recv)) {
		timeout_ = timeout_orig;
		return false;

	}

	timeout_ = timeout_orig;
	if (!CheckACK(recv,CMD_SET_BASIC, CMD_BASIC_DEV_BUILTIN_TEST)) 
    return false;
	if (recv[8]==0 && recv[9]==0 && recv[10]==0 && recv[11]==0) 
    return true;
	else {
		if (recv[8] & 0x1)  std::cerr << "AP-1: I2C Hardware Error." << std::endl;
		if (recv[8] & 0x2)  std::cerr << "AP-1: I2C EEPROM Error." << std::endl;
		if (recv[9] & 0x1)  std::cerr << "AHRS: Communication Error." << std::endl;
		if (recv[10] & 0x1) std::cerr << "GPS: Communication Error." << std::endl;
		if (recv[10] & 0x2) std::cerr << "GPS: 1PPS Signal Error." << std::endl;
		if (recv[10] & 0x4) std::cerr << "GPS: 1PPS Inhibit Error." << std::endl;
		if (recv[10] & 0x8) std::cerr << "GPS: Power Control Error." << std::endl;
		return false;
	}
}
bool Imu3DMGX3_45::SetAHRSMsgFormat() {

  std::vector<uint8_t> data;

  const uint8_t BYTE_SYNC1 = 0x75;
  const uint8_t BYTE_SYNC2 = 0x65;

	data.push_back(BYTE_SYNC1);
	data.push_back(BYTE_SYNC2);
	data.push_back(CMD_SET_3DM); // desc set
	data.push_back(0x0D); // length
	data.push_back(0x0D);
	data.push_back(CMD_3DM_AHRS_MSG_FORMAT);

	data.push_back(FUN_USE_NEW);
	data.push_back(0x03); // desc count

	data.push_back(0x04); // accelerometer vector
	data.push_back(0x0);
	data.push_back(100/rate_); // 20 Hz

	data.push_back(0x05); // gyro vector
	data.push_back(0x0);
	data.push_back(100/rate_); // 20 Hz

	data.push_back(0x0C); // euler angles
	data.push_back(0x0);
	data.push_back(100/rate_); // rate decimation -> 20 Hz

	AppendCRC(data);
	Write(data);
	WaitForMsg();

  std::vector<uint8_t> recv;
	size_t n = 8;
	recv = Read(n);

	if (!CheckCRC(recv)) 
    return false;
	if (!CheckACK(recv,CMD_SET_3DM,CMD_3DM_AHRS_MSG_FORMAT)) 
    return false;
	return true;
}

bool Imu3DMGX3_45::SetStream(uint8_t stream, bool state) {

  std::vector<uint8_t> data;

  const uint8_t BYTE_SYNC1 = 0x75;
  const uint8_t BYTE_SYNC2 = 0x65;

	data.push_back(BYTE_SYNC1);
	data.push_back(BYTE_SYNC2);
	data.push_back(CMD_SET_3DM);
	data.push_back(0x05);
	data.push_back(0x05);
	data.push_back(CMD_3DM_STREAM_STATE);
	data.push_back(0x1);
	data.push_back(stream);
	if (state) 
    data.push_back(0x01);
	else 
    data.push_back(0x0);

	AppendCRC(data);
	Write(data);
	WaitForMsg();

  std::vector<uint8_t> recv;
	size_t n = 8;
	recv = Read(n);

	if (!CheckCRC(recv)) {
		return false;
	}
	if (!CheckACK(recv,CMD_SET_3DM, CMD_3DM_STREAM_STATE)) {
    return false;
  }
	return true;
}

bool Imu3DMGX3_45::SendNoDataCmd(uint8_t cmd_set, uint8_t cmd) {

  std::vector<uint8_t> data;

  uint8_t BYTE_SYNC1 = 0x75;
  uint8_t BYTE_SYNC2 = 0x65;
  data.push_back(BYTE_SYNC1);
  data.push_back(BYTE_SYNC2);
  data.push_back(cmd_set); // desc set
  data.push_back(0x02); // length
  data.push_back(0x02);
  data.push_back(cmd);

  AppendCRC(data);
  Write(data);

  std::vector<uint8_t> recv;

  size_t n = 8;

  WaitForMsg();
  //cout << "do some reading..." << endl;
  recv = Read(n);
  //cout << "we have some data..." << endl;

  if (!CheckCRC(recv)) return false;
  if (!CheckACK(recv,cmd_set,cmd)) return false;
  return true;
}

bool Imu3DMGX3_45::PollAHRS(ImuObs& imuObs) {

  std::vector<uint8_t> data;

  const uint8_t BYTE_SYNC1 = 0x75;
  const uint8_t BYTE_SYNC2 = 0x65;
	data.push_back(BYTE_SYNC1);
	data.push_back(BYTE_SYNC2);
	data.push_back(CMD_SET_3DM); // desc set
	data.push_back(0x04); // length
	data.push_back(0x04);
	data.push_back(CMD_3DM_POLL_AHRS);
	data.push_back(0x1); // suppress ACK
	data.push_back(0x0);

	AppendCRC(data);
	Write(data);
	WaitForMsg();
  return ReceiveAHRS(imuObs);
}

bool Imu3DMGX3_45::ReceiveAHRS(ImuObs& imuObs) {

  std::vector<uint8_t> recv;
	size_t n = 46; 
	struct timespec curtime;
	clock_gettime(CLOCK_REALTIME, &curtime);

	recv = Read(n);

	if (!CheckCRC(recv)) 
    return false;

	//if (!checkACK(recv,CMD_SET_3DM,CMD_3DM_POLL_AHRS)) return false;
	// quaternion 0x0A, field length 18, MSB first
	//quat.time = posix_time::microsec_clock::local_time();

	if (recv[2] != 0x0E || recv[3] != 0x04) {
    std::cerr << "AHRS: Wrong msg format (0x04)." <<  std::endl;
		return false;
	}

	imuObs.t_host =  (int64_t)(curtime.tv_sec) * 1000000000 + (int64_t)(curtime.tv_nsec);
  
	imuObs.acc(0) = ExtractFloat(&recv[4]); // 0x04
	imuObs.acc(1) = ExtractFloat(&recv[4+4]);
	imuObs.acc(2) = ExtractFloat(&recv[4+8]);

	if (recv[16] != 0x0E || recv[17] != 0x05) {
    std::cerr << "AHRS: Wrong msg format (0x05)." << std::endl;
		return false;
	}

	imuObs.omega(0) = ExtractFloat(&recv[18]); // 0x05
	imuObs.omega(1) = ExtractFloat(&recv[18+4]);
	imuObs.omega(2) = ExtractFloat(&recv[18+8]);

	if (recv[30] != 0x0E || recv[31] != 0x0C) {
    std::cerr << "AHRS: Wrong msg format (0x0C)." << std::endl;
		return false;
	}

	 imuObs.rpy(0)= ExtractFloat(&recv[32]); // 0x0C
	 imuObs.rpy(1)= ExtractFloat(&recv[32+4]);
	 imuObs.rpy(2)= ExtractFloat(&recv[32+8]);

	/*quat.q0 = extractFloat(&recv[6]);
	quat.q1 = extractFloat(&recv[6+4]);
	quat.q2 = extractFloat(&recv[6+8]);
	quat.q3 = extractFloat(&recv[6+12]);*/
	//cout << quat.q0 << " " << quat.q1 << " " << quat.q2 << " " << quat.q3 << endl;
	return true;
}

void Imu3DMGX3_45::Read(char *data, size_t size)
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

void Imu3DMGX3_45::AppendCRC(std::vector<uint8_t>& arr) {
  uint8_t b1=0;
  uint8_t b2=0;
  for(size_t i=0; i<arr.size(); i++)
  {
    b1 += arr[i];
    b2 += b1;
  }
  arr.push_back(b1);
  arr.push_back(b2);
}

bool Imu3DMGX3_45::CheckCRC(std::vector<uint8_t>& arr) {
  uint8_t b1=0;
  uint8_t b2=0;

  if ( ((uint8_t)arr[1]+4) != (uint8_t)arr.size() ) {
    std::cout << "Sizes mismatch." << std::endl;
  }
  uint8_t end;
  if (((uint8_t)arr[1]+2) <= (uint8_t)arr.size()) 
    end = (uint8_t)arr[1]+2;
  else 
    end = (uint8_t)arr.size();

  b1 += BYTE_SYNC1;
  b2 += b1;
  b1 += BYTE_SYNC2;
  b2 += b1;

  for(size_t i=0; i<end; i++) {
    b1 += arr[i];
    b2 += b1;
  }

  /*for(unsigned int i=0; i<(arr.size()); i++)
    cout << static_cast<int>(arr[i]) << " ";
    cout << endl;*/

  if (b1==(uint8_t)arr[arr.size()-2] && b2==(uint8_t)arr[arr.size()-1]) 
    return true;
  else {
    std::cerr<< "Bad CRC." << std::endl;
    return false;
  }
}

bool Imu3DMGX3_45::CheckACK(std::vector<uint8_t>& arr, uint8_t cmd_set, uint8_t cmd) {
  if (arr.size() < 6) {
    std::cerr << "Too short reply." << std::endl;
    return false;
  }
  /*if (arr[0] != sync1 || arr[1] != sync2) {
    errMsg("Strange synchronization bytes.");
    return false;
    }*/
  if (arr[0] != cmd_set) {
    std::cerr << "Wrong desc set in reply." << std::endl;
    return false;
  }
  if (arr[4] != cmd) {
    std::cerr << "Wrong command echo." << std::endl;
    return false;
  }
  if (arr[5] != 0x0) {
    std::cerr << "NACK." << std::endl;
    return false;
  }
  return true;
}


}
