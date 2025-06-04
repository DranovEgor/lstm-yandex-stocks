import subprocess as sp
import sys


class FileRetriever:
    def __init__(self, directory_url: str):
        self.dir_link = directory_url

    def execute_shell(self, cmd_str: str, get_output: bool = False) -> str:
        """Handles shell command execution with error management"""
        try:
            cmd_result = sp.run(
                cmd_str,
                shell=True,
                executable="/bin/bash" if sys.platform != "win32" else None,
                check=True,
                text=True,
                stdout=sp.PIPE if get_output else None,
                stderr=sp.PIPE,
            )
            return cmd_result.stdout if get_output else None
        except sp.CalledProcessError as err:
            print(f"Command failed: {cmd_str}")
            print(f"Failure details: {err.stderr}")
            sys.exit(1)

    def setup_retrieval_tool(self):
        """Ensure required package is available"""
        print("Verifying gdown installation...")
        install_cmd = f"{sys.executable} -m pip install -U gdown"
        self.execute_shell(install_cmd)

    def acquire_resources(self):
        """Main method to obtain files"""
        self.setup_retrieval_tool()
        import gdown

        gdown.download_folder(
            url=self.dir_link,
            output="downloaded_content",
            quiet=False,
            remaining_ok=True,
        )


def init_process():
    resource_loader = FileRetriever(
        directory_url="https://drive.google.com/drive/folders/1uGKxUk-rrY7ae38PaPoEkXSsh-m4c0Sr?usp=sharing"
    )
    resource_loader.acquire_resources()


if __name__ == "__main__":
    init_process()
