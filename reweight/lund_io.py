"""
LUND I/O utilities.

LUND format used by this generator (per event):
  Header: nparticles  1 1 0 0 11 beam_energy target_pid target_mass 0
  Then nparticles lines, each:
    idx  lifetime  type  pid  parent  daughter  px py pz E  mass  vx vy vz
"""

from dataclasses import dataclass, field
from typing import List, Iterator, TextIO


@dataclass
class Particle:
    idx: int
    lifetime: float
    type: int
    pid: int
    parent: int
    daughter: int
    px: float
    py: float
    pz: float
    E: float
    mass: float
    vx: float
    vy: float
    vz: float

    def to_line(self) -> str:
        return (f"{self.idx} {self.lifetime:g} {self.type} {self.pid} "
                f"{self.parent} {self.daughter} "
                f"{self.px:.6f} {self.py:.6f} {self.pz:.6f} {self.E:.6f} "
                f"{self.mass:.6f} {self.vx:.6f} {self.vy:.6f} {self.vz:.6f}")


@dataclass
class Event:
    header: List[str]            # 10 tokens, kept as strings to preserve formatting
    particles: List[Particle] = field(default_factory=list)

    @property
    def nparticles(self) -> int:
        return int(self.header[0])

    @property
    def beam_energy(self) -> float:
        return float(self.header[6])

    def get(self, pid: int) -> Particle:
        """Return first particle matching pid."""
        for p in self.particles:
            if p.pid == pid:
                return p
        raise KeyError(f"pid {pid} not in event")

    def to_text(self) -> str:
        head = "\t" + " ".join(self.header)
        return head + "\n" + "\n".join(p.to_line() for p in self.particles) + "\n"


def read_lund(path: str) -> Iterator[Event]:
    """Yield Events from a LUND file one-by-one (memory-efficient)."""
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                return
            if not line.strip():
                continue
            header = line.split()
            n = int(header[0])
            particles = []
            for i in range(n):
                toks = f.readline().split()
                particles.append(Particle(
                    idx=int(toks[0]), lifetime=float(toks[1]), type=int(toks[2]),
                    pid=int(toks[3]), parent=int(toks[4]), daughter=int(toks[5]),
                    px=float(toks[6]), py=float(toks[7]), pz=float(toks[8]),
                    E=float(toks[9]), mass=float(toks[10]),
                    vx=float(toks[11]), vy=float(toks[12]), vz=float(toks[13]),
                ))
            yield Event(header=header, particles=particles)


def write_lund(path: str, events: Iterator[Event]) -> int:
    n = 0
    with open(path, "w") as f:
        for ev in events:
            f.write(ev.to_text())
            n += 1
    return n
