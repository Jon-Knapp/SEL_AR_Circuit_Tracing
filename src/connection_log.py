# connection_log.py
#
# The "brain" of the continuity-recording feature. It does four jobs:
#
#   1. Keeps the list of recorded connections (each is a pair of terminals that
#      were measured as electrically continuous).
#   2. Works out the GROUPS: which terminals are all connected together, even
#      indirectly. (Friend-group analogy below.)
#   3. Handles UNDO (remove the most recent connection).
#   4. Writes the records to a human-readable .txt table and to a SQLite
#      database for future queries.
#
# SINGLE SOURCE OF TRUTH
#   The list of connections is the only thing we truly store. Every time it
#   changes (a new connection, or an undo), we rebuild the groups, the .txt,
#   and the database FROM that list. This is a little wasteful, but the data is
#   tiny and it makes everything correct by construction: undo is just "remove
#   the last connection and rebuild", and a connection that merged two groups
#   correctly splits them apart again when undone.
#
# GROUPS, in plain terms (the "friend groups" idea)
#   If probe-pair A-B is connected, and B-C is connected, then A, B and C are
#   all in one group even though A and C were never directly touched together -
#   the same way that if Ann knows Bob and Bob knows Cara, the three of them
#   form one friend circle. Each group gets a label (G1, G2, ...) and a color
#   so it can be flagged on screen.
#
# LABEL NUMBERING (why you can see a gap like G1 and G3 with no G2)
#   Think of each NEW group taking a numbered ticket: the first group formed is
#   G1, the next is G2, and so on. When two groups merge, the merged group KEEPS
#   THE OLDER (lower) ticket number and the younger number is retired. So if G2
#   gets absorbed into G1, the number 2 is gone and you may see G1 and G3 side
#   by side. We chose stable labels (a group keeps its number for life) over
#   gap-free numbering, so flags don't reshuffle on screen mid-session.
#
# This module takes its settings (file paths, the group color palette) as plain
# arguments; it does NOT import config, matching object_detection.py's style.

import os
import sqlite3
from datetime import datetime


class ConnectionLog:
    """Records continuity connections, derives groups, and writes the files."""

    def __init__(self, txt_path, db_path, group_colors):
        """
        txt_path     : where to write the human-readable .txt table.
        db_path      : where to write the SQLite database.
        group_colors : list of (B, G, R) colors; group number N uses
                       group_colors[(N - 1) % len(group_colors)].
        """
        self.txt_path = txt_path
        self.db_path = db_path
        self.group_colors = group_colors

        # The one thing we actually store. Each item:
        #   {"timestamp": "...", "terminal_a": "...", "terminal_b": "..."}
        # with terminal_a <= terminal_b so A-B and B-A are the same pair.
        self.connections = []

        # Rebuilt from self.connections on every change.
        self.groups = []                 # list of group dicts (see below)
        self.group_for_terminal = {}     # terminal_id -> the group dict it is in

        self.session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build (empty) groups and write (empty) files now, so a record file
        # always exists and we confirm we can write to disk at startup.
        self._recompute_groups()
        self._write_files()

    # ------------------------------------------------------------------
    # Adding / removing connections
    # ------------------------------------------------------------------

    def add_connection(self, terminal_a, terminal_b):
        """
        Record a continuity connection between two terminals. Returns True if it
        was newly added, or False if it was ignored (same terminal twice, or a
        pair already on record). Rebuilds the groups and the files on success.
        """
        a, b = sorted([terminal_a, terminal_b])
        if a == b:
            return False                          # not a connection between two points

        for existing in self.connections:
            if existing["terminal_a"] == a and existing["terminal_b"] == b:
                return False                      # already recorded; do not duplicate

        self.connections.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "terminal_a": a,
            "terminal_b": b,
        })
        self._recompute_groups()
        self._write_files()
        return True

    def undo_last(self):
        """
        Remove the MOST RECENT connection and rebuild everything. Returns the
        removed connection dict, or None if there was nothing to undo.
        """
        if not self.connections:
            return None
        removed = self.connections.pop()
        self._recompute_groups()
        self._write_files()
        return removed

    # ------------------------------------------------------------------
    # Working out the groups (connected components, with stable labels)
    # ------------------------------------------------------------------

    def _recompute_groups(self):
        """
        Rebuild self.groups and self.group_for_terminal from self.connections.

        This uses "union-find": each terminal starts in its own little set, and
        every connection MERGES the two sets its terminals belong to. At the end,
        each remaining set is one group. We process the connections in the order
        they were recorded so the ticket-number rule (older number wins on a
        merge) comes out the same every time we rebuild.
        """
        parent = {}            # terminal -> its parent terminal (union-find tree)
        size = {}              # root terminal -> how many terminals are under it
        comp_label = {}        # root terminal -> its group number (or None)
        next_number = 0        # the next unused ticket number; only ever goes up

        def find(x):
            # Find the root of x's set, flattening the path as we go so future
            # look-ups are fast. (You don't need to follow the flattening to
            # understand it: find(x) returns "which set is x in".)
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        def ensure(x):
            # First time we see a terminal, put it in a set by itself.
            if x not in parent:
                parent[x] = x
                size[x] = 1
                comp_label[x] = None

        for connection in self.connections:
            a = connection["terminal_a"]
            b = connection["terminal_b"]
            ensure(a)
            ensure(b)
            root_a = find(a)
            root_b = find(b)

            if root_a == root_b:
                # Already in the same group (e.g. A-C measured after A-B and
                # B-C). It is still a real measurement we keep in the list, but
                # it forms no new group and takes no new ticket.
                continue

            label_a = comp_label[root_a]
            label_b = comp_label[root_b]
            known_labels = [label for label in (label_a, label_b)
                            if label is not None]
            if known_labels:
                merged_label = min(known_labels)   # merge KEEPS the older number
            else:
                next_number += 1                   # a brand-new group takes a ticket
                merged_label = next_number

            # Merge the two sets. We attach the smaller tree under the larger
            # one (just keeps the trees shallow); the survivor carries the label.
            if size[root_a] < size[root_b]:
                root_a, root_b = root_b, root_a
            parent[root_b] = root_a
            size[root_a] += size[root_b]
            comp_label[root_a] = merged_label
            comp_label.pop(root_b, None)

        # Turn the union-find result into a tidy list of group dicts.
        members_by_number = {}
        for terminal in parent:
            number = comp_label[find(terminal)]
            members_by_number.setdefault(number, set()).add(terminal)

        self.groups = []
        self.group_for_terminal = {}
        for number in sorted(members_by_number):
            color = self.group_colors[(number - 1) % len(self.group_colors)]
            group = {
                "number": number,
                "label": f"G{number}",
                "color": color,
                "terminal_ids": members_by_number[number],
            }
            self.groups.append(group)
            for terminal in group["terminal_ids"]:
                self.group_for_terminal[terminal] = group

    # ------------------------------------------------------------------
    # Writing the record files
    # ------------------------------------------------------------------

    def _write_files(self):
        self._write_txt()
        self._write_sqlite()

    def _write_txt(self):
        """Rewrite the whole .txt table from the current connections + groups."""
        # Make the terminal columns wide enough for the longest id we have.
        id_width = len("terminal_a")
        for connection in self.connections:
            id_width = max(id_width, len(connection["terminal_a"]),
                           len(connection["terminal_b"]))

        lines = [
            "Continuity connection log",
            f"Session started: {self.session_start}",
            "This file is rebuilt automatically whenever a connection is added "
            "or undone.",
            "",
            f'{"timestamp".ljust(19)}  {"terminal_a".ljust(id_width)}  '
            f'{"terminal_b".ljust(id_width)}  group',
        ]
        lines.append("-" * len(lines[-1]))

        for connection in self.connections:
            group = self.group_for_terminal.get(connection["terminal_a"])
            label = group["label"] if group else "-"
            lines.append(
                f'{connection["timestamp"].ljust(19)}  '
                f'{connection["terminal_a"].ljust(id_width)}  '
                f'{connection["terminal_b"].ljust(id_width)}  {label}'
            )

        lines.append("")
        lines.append("Groups (all terminals electrically connected together):")
        if self.groups:
            for group in self.groups:
                members = ", ".join(sorted(group["terminal_ids"]))
                lines.append(f'  {group["label"]}: {members}')
        else:
            lines.append("  (none yet)")
        lines.append("")

        with open(self.txt_path, "w") as file:
            file.write("\n".join(lines))

    def _write_sqlite(self):
        """
        Rewrite the SQLite database from the current state. Two tables:
          connections     - the actual measured pairs (the faithful record).
          terminal_groups - the derived "which group is each terminal in".
        We clear and refill both each time, so the database always matches the
        current set of connections (including after an undo).
        """
        connection = sqlite3.connect(self.db_path)
        try:
            cursor = connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS connections (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp  TEXT,
                    terminal_a TEXT,
                    terminal_b TEXT,
                    group_id   TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS terminal_groups (
                    terminal_id TEXT PRIMARY KEY,
                    group_id    TEXT
                )
            """)
            cursor.execute("DELETE FROM connections")
            cursor.execute("DELETE FROM terminal_groups")

            for record in self.connections:
                group = self.group_for_terminal.get(record["terminal_a"])
                label = group["label"] if group else None
                cursor.execute(
                    "INSERT INTO connections "
                    "(timestamp, terminal_a, terminal_b, group_id) "
                    "VALUES (?, ?, ?, ?)",
                    (record["timestamp"], record["terminal_a"],
                     record["terminal_b"], label),
                )

            for terminal_id, group in self.group_for_terminal.items():
                cursor.execute(
                    "INSERT INTO terminal_groups (terminal_id, group_id) "
                    "VALUES (?, ?)",
                    (terminal_id, group["label"]),
                )

            connection.commit()
        finally:
            connection.close()