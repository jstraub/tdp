
#include <tdp/inertial/imu_outstream.h>
#include <pangolin/log/packetstream.h>
#include <pangolin/utils/file_utils.h>
#include <pangolin/utils/uri.h>
#include <pangolin/utils/sigstate.h>
#include <set>

#ifndef _WIN_
#  include <unistd.h>
#endif


namespace tdp {

void SigPipeHandler(int sig)
{
  pangolin::SigState::I().sig_callbacks.at(sig).value = true;
}

ImuOutStream::ImuOutStream(const std::string& path, size_t buffer_size_bytes) 
  : filename(path), 
  packetstream_buffer_size_bytes(buffer_size_bytes),
  packetstreamsrcid(-1),
  first_frame(true),
  is_pipe(pangolin::IsPipe(filename))
{
  if(!is_pipe)
  {
    packetstream.Open(filename, packetstream_buffer_size_bytes);
  }
  else
  {
    pangolin::RegisterNewSigCallback(&SigPipeHandler, (void*)this, SIGPIPE);
  }
}

ImuOutStream::~ImuOutStream() 
{}

bool ImuOutStream::IsOpen() const {
}

void ImuOutStream::Open(const std::string& uri, const
    pangolin::json::value& properties) {

  if(packetstreamsrcid == -1) {
    input_uri = uri;
    device_properties = properties;

    pangolin::json::value json_header(pangolin::json::object_type,false);
    pangolin::json::value& json_streams = json_header["streams"];
    json_header["device"] = device_properties;

    pangolin::json::value& json_stream = json_streams.push_back();
    json_stream["version"] = ImuObs::VERSION;
    json_stream["bytes"] =   sizeof(ImuObs);
    json_stream["offset"] =  0;

    total_frame_size = sizeof(ImuObs);

    pangolin::PacketStreamSource pss = packetstream.CreateSource(
        "imu_obs", input_uri, json_header,
        total_frame_size,
        "struct ImuObs{"
        "Eigen::Vector3f acc; // acceleration in m/s^2"
        "Eigen::Vector3f omega; // rotational velocity in rad/s"
        "Eigen::Vector3f rpy; // magnetometer in ?"
        "Eigen::Vector3f mag; // magnetometer in ?"
        "int64_t t_host;"
        "int64_t t_device;"
        "static const int VERSION = 1;"
        "};"
        );
    packetstreamsrcid = (int)pss.id;
    packetstream.AddSource(pss);
  }else{
    throw std::runtime_error("Unable to add new streams");
  }
}
void ImuOutStream::Close() {

}

int ImuOutStream::WriteStream(const ImuObs& data, 
      const pangolin::json::value& frame_properties) {
#ifndef _WIN_
    if(is_pipe)
    {
        // If there is a reader waiting on the other side of the pipe, open
        // a file descriptor to the file and close it only after the file
        // has been opened by the PacketStreamWriter. This avoids the reader
        // from seeing EOF on its next read because all file descriptors on
        // the write side have been closed.
        //
        // When the stream is already open but the reader has disappeared,
        // opening a file descriptor will fail and errno will be ENXIO.
        int fd = pangolin::WritablePipeFileDescriptor(filename);

        if (!packetstream.IsOpen()) {
            if (fd != -1) {
                packetstream.Open(filename, packetstream_buffer_size_bytes);
                close(fd);
                first_frame = true;
            }
        } else {
            if (fd != -1) {
                // There's a reader on the other side of the pipe.
                close(fd);
            } else {
                if (errno == ENXIO) {
                    packetstream.ForceClose();
                    pangolin::SigState::I().sig_callbacks.at(SIGPIPE).value
                      = false;
                    // This should be unnecessary since per the man page,
                    // data should be dropped from the buffer upon closing the
                    // writable file descriptors.
                    pangolin::FlushPipe(filename);
                }
            }
        }

        if(!packetstream.IsOpen())
        {
            return 0;
        }
    }
#endif

    if(first_frame)
    {
        first_frame = false;
        packetstream.WriteSources();
    }

    if(!frame_properties.is<pangolin::json::null>()) {
        packetstream.WriteSourcePacketMeta(packetstreamsrcid, frame_properties);
    }

    packetstream.WriteSourcePacket(
        packetstreamsrcid,
        (char*)&data, total_frame_size);

    return 0;
}

}
