import math

class PerviousConcreteBatchConverter:
    def __init__(self):
        self.chockalingam_coefficients = {
            'cube': {'B': 4.31, 'C': -2.03, 'Da': 150},
            'cylinder': {'B': 3.13, 'C': -1.19, 'Da': 300}
        }
        self.cube_to_cylinder_ratio = 1 / 1.12  # cube = 1.12 × cylinder

        self.STD_DIAMETER = 100
        self.STD_HEIGHT = 200

    def msel_term(self, D, shape_type):
        coeff = self.chockalingam_coefficients[shape_type]
        B = coeff['B']
        C = coeff['C']
        Da = coeff['Da']
        return B / math.sqrt(1 + D / Da) + C

    def cube_to_std_cube(self, strength, side):
        if side == 150:
            return strength
        term = self.msel_term(side, 'cube')
        return strength / term

    def cube_from_std_cube(self, sigma_std, side):
        if side == 150:
            return sigma_std
        term = self.msel_term(side, 'cube')
        return sigma_std * term

    def cyl_to_std_cyl(self, strength, dia, height):
        if dia == 150 and height == 300:
            return strength
        term = self.msel_term(height, 'cylinder')
        return strength / term

    def cyl_from_std_cyl(self, sigma_std, dia, height):
        if dia == 150 and height == 300:
            return sigma_std
        term = self.msel_term(height, 'cylinder')
        return sigma_std * term

    def convert_to_100x200(self, strength, in_type, d1, d2):
        if in_type == 'cylinder' and d1 == 100 and d2 == 200:
            return strength

        if in_type == 'cube':
            cube_std = self.cube_to_std_cube(strength, d1)
            cyl_std = cube_std * self.cube_to_cylinder_ratio
        else:
            cyl_std = self.cyl_to_std_cyl(strength, d1, d2)
            cube_std = cyl_std / self.cube_to_cylinder_ratio

        result = self.cyl_from_std_cyl(cyl_std, self.STD_DIAMETER, self.STD_HEIGHT)
        return result

    def interactive(self):
        print("=== Compressive Strength Converter → 100×200 mm Cylinder (Excel Paste Supported) ===")

        shape = input("Enter input specimen shape (cube/cylinder): ").strip().lower()
        if shape == "cube":
            d1 = float(input("Enter cube side length (mm): ").strip())
            d2 = d1
        else:
            d1 = float(input("Enter cylinder diameter (mm): ").strip())
            d2 = float(input("Enter cylinder height (mm): ").strip())

        print("\nPaste strengths from Excel and press Enter on an empty line:\n")
        pasted = []
        while True:
            line = input()
            if line.strip() == "":
                break
            pasted.append(line.strip())

        data = " ".join(pasted).replace(",", " ").split()
        strengths = [float(x) for x in data]

        print("\n=== Converted Strengths (100×200 mm Cylinder) ===")
        for s in strengths:
            try:
                converted = self.convert_to_100x200(s, shape, d1, d2)
                print(f"{converted:.3f}")
            except:
                print("Conversion error")

        print("\nBatch conversion complete.")

if __name__ == "__main__":
    PerviousConcreteBatchConverter().interactive()

