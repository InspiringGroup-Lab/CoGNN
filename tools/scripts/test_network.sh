ip netns exec B iperf -s -p 8001
ip netns exec A iperf -c 10.0.0.2 -p 8001 -t 10 -i 2