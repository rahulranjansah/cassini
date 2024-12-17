# HDFS Setup and Usage

## Prerequisites

- Hadoop installed and configured
- HDFS services running

## Starting HDFS Services

To start HDFS services, run the following commands:

```bash
# Start HDFS
start-dfs.sh

# Start YARN
start-yarn.sh

# Creating Directories in HDFS

hdfs dfs -mkdir /user/yourusername/input
hdfs dfs -mkdir /user/yourusername/output

# Upload dataset to HDFS
hdfs dfs -put /path/to/local/dataset /user/yourusername/input