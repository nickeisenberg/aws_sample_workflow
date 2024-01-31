from shwrap.transfer import SecureCopyProtocol
import os

scp = SecureCopyProtocol(
    user="nick",
    ip="50.18.67.101",
    port="22",
    pem=os.environ["USWEST1"]
)

source_path = "/home/nicholas/GitRepos/aws_sample_workflow/Network/main.py"
save_path = "/nvme1n1users/nick/Experiments/aws_workflow_exp"

scp.put(source_path=source_path, save_path=save_path)
