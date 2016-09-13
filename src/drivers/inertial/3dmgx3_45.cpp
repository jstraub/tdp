
#include <tdp/drivers/inertial/3dmgx3_45.h>

namespace tdp {

//Imu3DMGX3_45::Imu3DMGX3_45() {
//}

//Imu3DMGX3_45::~Imu3DMGX3_45() {
//}

void Imu3DMGX3_45::GrabNext(ImuObs& obs) {

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
  serial_.set_option(asio::serial_port_base::baud_rate(19200));
  serial_.set_option(PARITY);
  serial_.set_option(CHAR_SIZE);
  serial_.set_option(FLOW_CTRL);
  serial_.set_option(STOP);

  if (SendNoDataCmd(CMD_SET_BASIC, CMD_BASIC_PING)) {
    std::cout << "ping successfull " << std::endl;
  }


}

void Imu3DMGX3_45::Stop() {
  if (!serial_.is_open())
    return;
  serial_.close();
}

bool Imu3DMGX3_45::SendNoDataCmd(uint8_t cmd_set, uint8_t cmd) {

  std::vector<uint8_t> data;

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

  waitForMsg();
  //cout << "do some reading..." << endl;
  recv = Read(n);
  //cout << "we have some data..." << endl;

  if (!CheckCRC(recv)) return false;
  if (!CheckACK(recv,cmd_set,cmd)) return false;
  return true;
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
