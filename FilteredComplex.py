import logging
import os
import sys

path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir,
                                    os.pardir))
sys.path.append(os.path.join(path, "Dionysus-python3/build/bindings/python"))


class FilteredComplex:

    """Class abstracting geometric/algebraic filtration of a complex;
    (fcomplex should be dionysus.Filtration() object).
    Attributes:
    h_0                          pairs of points on the diagram
    persistence_diagram_0        the diagram (dionysus object)
    h_1                          (uncomment, if needed)
    persistence_diagram_1        (uncomment, if needed)
    bottleneck_dist              distance of 0-diagram and the empty diagram

    Methods:
    compute_persistence_pairs
    create_persistence_diagrams
    distance
    """
    dionysus = __import__('dionysus')
    empty_diagram = dionysus.PersistenceDiagram(0)
    empty_diagram.append((0, 0))

    def __init__(self, fcomplex):

        logging.debug("Initialising Static Presistence")
        self.homology_0 = {}
        self.persistence = self.dionysus.StaticPersistence(fcomplex)

        logging.debug("Pairing Simplices")
        self.persistence.pair_simplices()
        self.smap = self.persistence.make_simplex_map(fcomplex)

    def distance(self, persistence_diagram, p=0):
        """Returns p-th Wasserstein distance between the filtration's 0-diagram
        and the empty diagram.
        If p=0 then the bottleneck is returned.

        TODO: higher dimensions"""

        if p > 0:
            return self.dionysus.wasserstein_distance(
                persistence_diagram, self.empty_diagram, p)
        else:
            return self.dionysus.bottleneck_distance(
                persistence_diagram, self.empty_diagram)

    def compute_homology(self, dimension=0):
        """ Computes the homology persistence pairs for the filtered complex"""

        logging.debug("Computing 0-th Persistence Homology Pairs")

        homology = {"dying": [], "undying": []}
        undying = 0
        for i in self.persistence:
            if i.sign() > 0:
                birth_simplex = self.smap[i]
                birth = birth_simplex.data
                if birth_simplex.dimension() == dimension:
                    if i.unpaired():
                        death = float('inf')
                        homology["undying"].append([birth, death])
                        logging.debug("Undying simplex: %s at %f",
                                      birth_simplex, birth_simplex.data)
                        undying += 1
                        if undying > 1:
                            logging.debug("The complex seems to be "
                                          "disconnected?!")
                    else:
                        killing_simplex = self.smap[i.pair()]
                        death = killing_simplex.data
                        if death > birth:
                            homology["dying"].append([birth, death])
                        elif death < birth:
                            logging.warning("You can not die before You were "
                                            "born!")
                            logging.warning(birth_simplex, birth,
                                            killing_simplex, death)
                elif birth_simplex.dimension() > 1:
                    logging.warning("There should be no simplices of dim >1?! "
                                    "but there is: %s", birth_simplex)
        return homology

    def create_persistence_diagrams(self):
        logging.debug("Creating Diagrams")

        if self.homology_0["dying"]:
            persistence_diagram = self.dionysus.PersistenceDiagram(0)
            all_pairs = [tuple(x) for x in self.homology_0["dying"]]
            for x in all_pairs:
                persistence_diagram.append(x)
        else:
            persistence_diagram = self.empty_diagram

        return persistence_diagram


class SweepFilteredComplex(FilteredComplex):

    def __init__(self, fcomplex, axis, inverse):
        self.axis = axis
        self.inverse = int(inverse)

        FilteredComplex.__init__(self, fcomplex)

        self.homology_0 = self.compute_homology(dimension=0)

    def compute_homology(self, dimension=0):
        logging.debug("Computing 0-th Persistence Homology Pairs")

        homology = {"dying": [], "undying": []}
        undying = 0
        for i in self.persistence:
            if i.sign() > 0:
                birth_simplex = self.smap[i]
                birth = birth_simplex.data[self.inverse][self.axis]
                if birth_simplex.dimension() == dimension:
                    if i.unpaired():
                        death = float('inf')
                        homology["undying"].append([birth, death])
                        logging.debug("Undying simplex: %s at %f",
                                      birth_simplex, birth)
                        undying += 1
                        if undying > 1:
                            logging.warning("The complex seems to be "
                                            "disconnected?!")
                    else:
                        killing_simplex = self.smap[i.pair()]
                        death = killing_simplex.data[self.inverse][self.axis]
                        if death > birth:
                            homology["dying"].append([birth, death])
                        elif death < birth:
                            logging.warning("You can not die before You were "
                                            "born!")
                            logging.warning(birth_simplex, birth,
                                            killing_simplex, death)
                elif birth_simplex.dimension() > 1:
                    logging.warning("There should be no simplices of dim >1?! "
                                    "but there is: %s", birth_simplex)
        return homology


class AlphaFilteredComplex(FilteredComplex):

    def __init__(self, fcomplex):
        FilteredComplex.__init__(self, fcomplex)

        self.homology_0 = self.compute_homology(dimension=0)

    def compute_homology(self, dimension=0):
        logging.debug("Computing 0-th Persistence Homology Pairs")

        homology = {"dying": [], "undying": []}
        undying = 0
        for i in self.persistence:
            if i.sign() > 0:
                birth_simplex = self.smap[i]
                birth = birth_simplex.data[0]
                if birth_simplex.dimension() == dimension:
                    if i.unpaired():
                        death = float('inf')
                        homology["undying"].append([birth, death])
                        logging.debug("Undying simplex: %s at %f",
                                      birth_simplex, birth)
                        undying += 1
                        if undying > 1:
                            logging.warning("The complex seems to be "
                                            "disconnected?!")
                    else:
                        killing_simplex = self.smap[i.pair()]
                        death = killing_simplex.data[0]
                        if death > birth:
                            homology["dying"].append([birth, death])
                        elif death < birth:
                            logging.warning("You can not die before You were "
                                            "born!")
                            logging.warning(birth_simplex, birth,
                                            killing_simplex, death)
                elif birth_simplex.dimension() > 1:
                    logging.warning("There should be no simplices of dim >1?! "
                                    "but there is: %s", birth_simplex)
        return homology
