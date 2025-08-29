import sys
import subprocess


def main():
    """
    Launcher script for Streamlit app.
    Just click ▶ Run in PyCharm on this file to start your app.
    """
    script_to_run = "app_ui.py"  # The Streamlit UI file

    # This runs: python -m streamlit run app_ui.py
    subprocess.run([sys.executable, "-m", "streamlit", "run", script_to_run])


if __name__ == "__main__":
    main()
