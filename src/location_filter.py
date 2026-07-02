class LocationFilter:
    def __init__(self, user_locations):
        """
        user_locations: list of locations the user can attend, e.g. ["Toronto", "Jeddah", "Online"]
        Matching is case-insensitive and checks for partial matches
        (so "Toronto" matches "Toronto, Canada" or "Online" matches "Fully Online").
        """
        self.user_locations = [loc.lower().strip() for loc in user_locations]

    def passes(self, event_location):
        """
        Returns True if the event's location matches one of the user's
        acceptable locations, False otherwise.
        """
        if not event_location:
            return False

        event_location = event_location.lower().strip()

        for user_loc in self.user_locations:
            if user_loc in event_location or event_location in user_loc:
                return True

        return False


if __name__ == "__main__":
    filt = LocationFilter(["Toronto", "Jeddah", "Online"])

    test_cases = [
        "Online",
        "Toronto, Canada",
        "Jeddah, Saudi Arabia",
        "San Francisco, CA",
        "New York",
        "",
    ]

    print("Testing LocationFilter:\n")
    for loc in test_cases:
        result = filt.passes(loc)
        print(f"  '{loc}' -> {result}")