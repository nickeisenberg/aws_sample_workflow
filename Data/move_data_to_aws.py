from shwrap.transfer import SecureCopyProtocol
import os

scp = SecureCopyProtocol(
    user="nick",
    ip="50.18.67.101",
    port="22",
    pem=os.environ["USWEST1"]
)

data_path = "/home/nicholas/Datasets/randn"

scp.put(
    source_path=data_path,
    save_path="/nvme2n1/Datasets"
)
