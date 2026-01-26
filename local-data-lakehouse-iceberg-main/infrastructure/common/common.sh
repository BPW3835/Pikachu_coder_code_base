#!/bin/bash

download_and_verify() {
  # =======================================================================
  # Load the input parameters
  # =======================================================================
  local jar_url=$1
  local md5_url=$2
  local download_dir=$3

  # =======================================================================
  # Set up jar and md5 file names
  # =======================================================================
  local jar_file=$(basename "${jar_url}")
  local md5_file="${jar_file}.md5"
  
  # =======================================================================
  # Download jar and md5 files
  # =======================================================================
  echo "Downloading ${jar_file} to ${download_dir}/packages"

  # If md5 file doesn't exist, then download it
  if [ ! -f "${download_dir}/packages/${md5_file}" ]; then
    curl -L -o "${download_dir}/packages/${md5_file}" "${md5_url}"
  fi

  # If jar file doesn't exist, then download it
  if [ ! -f "${download_dir}/packages/${jar_file}" ]; then
    curl -L -o "${download_dir}/packages/${jar_file}" "${jar_url}"
  fi

  # =======================================================================
  # Verify the checksum of the downloaded jar file
  # =======================================================================
  # Determine the md5 command to use based on the operating system
  local md5_command
  if [[ "$(uname)" == "Darwin" ]]; then
    # macOS
    md5_command="md5 -q"
  elif [[ "$(uname)" == "Linux" ]]; then
    md5_command="md5sum"
  else
    break
  fi

  # Compute jar file md5
  local_md5=$($md5_command "${download_dir}/packages/${jar_file}" | awk '{ print $1 }')
  # Get md5 from md5 file
  file_md5=$(cat "${download_dir}/packages/${md5_file}")

  # Checksum verification (compare local md5 with file md5)
  if [ "${local_md5}" != "${file_md5}" ]; then
    echo "Use ${md5_file} to MD5 checksum ${jar_file} verification failed, Please delete it."
    exit 1
  fi
}