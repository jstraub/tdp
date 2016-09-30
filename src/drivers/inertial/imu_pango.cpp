#include <tdp/inertial/imu_obs.h>
#include <tdp/drivers/inertial/imu_pango.h>
#include <pangolin/utils/file_utils.h>

#include <pangolin/utils/file_utils.h>
#include <pangolin/compat/bind.h>

#ifndef _WIN_
#  include <unistd.h>
#endif

namespace tdp {

ImuPango::ImuPango(const std::string& filename, bool realtime)
  : reader(filename, realtime), filename(filename), realtime(realtime),
  is_pipe(pangolin::IsPipe(filename)),
  is_pipe_open(true),
  pipe_fd(-1)
{
  // N.B. is_pipe_open can default to true since the reader opens the file and
  // reads header information from it, which means the pipe must be open and
  // filled with data.
  src_id = 0; //FindSource();
  size_bytes = sizeof(ImuObs);

}

ImuPango::~ImuPango()
{
#ifndef _WIN_
    if (pipe_fd != -1) {
      close(pipe_fd);
    }
#endif
}

bool ImuPango::GrabNext(ImuObs& obs) {
#ifndef _WIN_
    if (is_pipe && !is_pipe_open) {
        if (pipe_fd == -1) {
            pipe_fd = pangolin::ReadablePipeFileDescriptor(filename);
        }

        if (pipe_fd == -1) {
            return false;
        }

        // Test whether the pipe has data to be read. If so, open the
        // file stream and start reading. After this point, the file
        // descriptor is owned by the reader.
        if (pangolin::PipeHasDataToRead(pipe_fd)) {
            reader.Open(filename, realtime);
            close(pipe_fd);
            is_pipe_open = true;
        } else {
            return false;
        }
    }
#endif

    try {
        if(reader.ReadToSourcePacketAndLock(src_id)) {
            // read this frames actual data
            reader.Read((char*)&obs, size_bytes);
            reader.ReleaseSourcePacketLock(src_id);
            return true;
        } else {
            if (is_pipe && !reader.stream().good()) {
                HandlePipeClosed();
            }
            return false;
        }
    } catch (std::exception& ex) {
        if (is_pipe) {
            HandlePipeClosed();
            return false;
        } else {
            throw ex;
        }
    }
}

bool ImuPango::GrabNewest(ImuObs& obs) {
  return GrabNext(obs);
}

void ImuPango::Stop() {
}

void ImuPango::Start() {
}

pangolin::json::value ImuPango::GetProperties() const {

  if(src_id >=0) {
    return reader.Sources()[src_id].info["device"];
  }else{
    throw std::runtime_error("Not initialised");
  }
}

pangolin::json::value ImuPango::GetFrameProperties() const
{
    if(src_id >=0) {
        return reader.Sources()[src_id].meta;
    }else{
        throw std::runtime_error("Not initialised");
    }
}

void ImuPango::HandlePipeClosed()
{
    // The pipe was closed by the other end. The pipe will have to be
    // re-opened, but it is not desirable to block at this point.
    //
    // The next time a frame is grabbed, the pipe will be checked and if
    // it is open, the stream will be re-opened.
    reader.Close();
    is_pipe_open = false;
}

}
