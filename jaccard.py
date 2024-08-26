def jaccard_similarity(text1, text2):
    # Tokenize the texts into words
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())

    # Calculate the intersection and union of the two sets
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    # Calculate the Jaccard similarity
    jaccard_sim = len(intersection) / len(union)
    
    return jaccard_sim

# Example usage:
text1 = "[0095] In an embodiment, the Transverse Anderson Localization energy relays areconfigured as loose coherent or flexible energy relay elements. Considerations for 4D Plenoptic Functions:Selective Propagation of Energy through Holographic Waveguide Arrays"
text2 = "20. An optical system, comprising: a fore optic; a first set of lenslets configured to receive incoming rays from the fore optic and decompose an image within a field of view into a plurality of multi-pixel subfields by rotation of a chief ray within each multi-pixel subfield by a differing amount relative to a rotation of other rays of the subfield, the first set of lenslets configured to select pixels for each subfield based on angle of incoming rays of the image; a spatial light modulator disposed in an optical path after the first set of lenslets, the spatial light modulator including a plurality of spatial light modulation elements; a controller configured to cause the spatial light modulation elements to selectively block or pass light from at least a portion of one or more of the subfields and selectively modulate light from at least a portion of one of the subfields to provide for selection and a spatial-temporal encoding of the subfields; a second set of lenslets configured to provide the portions of the subfields that are passed onto substantially overlapping areas of a common image plane; and a plurality of achromatic prisms configured to compress an angle space of the chief rays of each of the subfields to provide a more-parallel alignment of the chief rays prior to subfield image formation by the second set of lenslets."

print(jaccard_similarity(text1, text2))