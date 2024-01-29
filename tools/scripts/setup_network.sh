# Create the bridge
ip link add br0 type bridge
ip link set dev br0 up

# Create network namespaces and veth interfaces and plug them into the bridge
for host in {A..E} ; do 
    ip netns add ${host}
    ip link add veth${host} type veth peer name veth${host}.peer
    ip link set dev veth${host}.peer master br0
    ip link set dev veth${host}.peer up
    ip link set dev veth${host} netns ${host}
    ip netns exec ${host} ip link set dev veth${host} up
    ip netns exec ${host} ip link set dev lo up
done

# Assign IPs
ip addr add 10.0.0.254/24 dev br0

COUNTER=0
for host in {A..E} ; do
    COUNTER=$((COUNTER+1))
    echo COUNTER: ${COUNTER}
    ip netns exec ${host} ip addr add 10.0.0.${COUNTER}/24 dev veth${host}
    ip netns exec ${host} ip route add default via 10.0.0.${COUNTER} dev veth${host}
done

# ip netns exec B ip addr add 10.0.0.2/24 dev vethB
# ip netns exec C ip addr add 10.0.0.3/24 dev vethC
# ip netns exec D ip addr add 10.0.0.4/24 dev vethD
# ip netns exec E ip addr add 10.0.0.5/24 dev vethE

for host in {A..E} ; do
    # ip netns exec ${host} tc qdisc del dev veth${host}
    ip netns exec ${host} tc qdisc add dev veth${host} root handle 1: htb default 11
    ip netns exec ${host} tc class add dev veth${host} parent 1: classid 1:1 htb rate 10000Mbps
    ip netns exec ${host} tc class add dev veth${host} parent 1:1 classid 1:11 htb rate ${1}Mbit
    ip netns exec ${host} tc qdisc add dev veth${host} parent 1:11 handle 10: netem delay ${2}ms
    ip netns exec ${host} tc qdisc list
done

iptables -P FORWARD ACCEPT

