for host in {A..E} ; do 
    ip netns exec ${host} tc qdisc del dev veth${host} root
done

# Remove veth pairs and network namespaces
for host in {A..E} ; do
    ip link del dev veth${host}.peer
    ip netns del ${host}
done

# Remove the bridge
ip link del dev br0

iptables -P FORWARD DROP