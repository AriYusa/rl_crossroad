1.`docker build -t ros_noetic:1 .`
2. Install some X client (for example, XLaunch for Windows) and run it (ask GPT what settings to choose)
3. Check your ip (172.xxx....) and run
`docker run -it --rm -e DISPLAY=<your_ip>:0 ros_noetic:1`
example `docker run -it --rm -e DISPLAY=172.19.192.1:0 ros_noetic:1`
4. Inside docker contailner run `roslaunch jackal_robot crossroad.launch`