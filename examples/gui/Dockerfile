# Minimal desktop image for the `gui` task and the gui_* tools: an X server, an input
# driver, a screenshot tool, and a tiny Tk test app. Build and use as the sandbox image.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive DISPLAY=:0
RUN apt-get update \
 && apt-get install -y --no-install-recommends xvfb xdotool imagemagick x11-utils python3 python3-tk \
 && rm -rf /var/lib/apt/lists/*

COPY button_app.py /opt/gui/button_app.py
COPY start_desktop.sh /opt/gui/start_desktop.sh
RUN chmod +x /opt/gui/start_desktop.sh

# Starts Xvfb on $DISPLAY (1024x768) and waits until it accepts connections.
ENTRYPOINT ["/opt/gui/start_desktop.sh"]
CMD ["sleep", "infinity"]
