#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Element {
    H,
    C,
    N,
    O,
    /// Phosphorus, used in nucleic-acid backbones. Added when
    /// `chem::nucleotide` was introduced; protein-only code paths
    /// continue to ignore it.
    P,
    S,
}

impl Element {
    pub const fn symbol(self) -> char {
        match self {
            Element::H => 'H',
            Element::C => 'C',
            Element::N => 'N',
            Element::O => 'O',
            Element::P => 'P',
            Element::S => 'S',
        }
    }

    /// Average atomic mass in Daltons.
    pub const fn mass_da(self) -> f64 {
        match self {
            Element::H => 1.008,
            Element::C => 12.011,
            Element::N => 14.007,
            Element::O => 15.999,
            Element::P => 30.974,
            Element::S => 32.06,
        }
    }

    /// Standard van der Waals radius in Å (Bondi 1964).
    pub const fn vdw_radius_a(self) -> f64 {
        match self {
            Element::H => 1.20,
            Element::C => 1.70,
            Element::N => 1.55,
            Element::O => 1.52,
            Element::P => 1.80,
            Element::S => 1.80,
        }
    }
}
