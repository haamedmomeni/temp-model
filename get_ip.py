import netifaces as ni


def get_wired_interface_ip():
    """
    Function to retrieve the IP address of the wired network interface of the current machine.
    """
    try:
        # Get a list of all available network interfaces on the system
        interfaces = ni.interfaces()

        # List of keywords to identify wireless interfaces
        wireless_keywords = []  # ["wl", "wlan", "wifi", "wireless"]
        # Loop through each interface to find the wired one
        for interface in interfaces:
            # Exclude loopback ('lo') interfaces
            if "lo" not in interface and\
                    not any(keyword in interface.lower() for keyword in wireless_keywords):
                # Get addresses associated with the current interface
                addresses = ni.ifaddresses(interface)

                # Check if there are addresses available for the interface
                if ni.AF_INET in addresses:
                    # Retrieve the IPv4 address from the interface
                    ip_address = addresses[ni.AF_INET][0]['addr']
                    # Return the IP address of the first non-loopback wired interface found
                    return ip_address

        # Raise a custom exception if no wired interface is found
        raise Exception("No wired interface found")

    except Exception as e:
        print(f"Error getting wired interface IP: {e}")
