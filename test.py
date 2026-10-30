"""Testes do ft_linear_regression. Rodar com: python3 test.py

Os imports ficam dentro de cada teste de propósito: se predict.py ainda não
existir (ou estiver quebrado), só os testes dele falham — os do data.py rodam.
"""
import os
import sys
import tempfile
import unittest

# O código de produção mora em src/: põe essa pasta no caminho de import.
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC)
DATA = os.path.join(SRC, "data.csv")


class TestLoadData(unittest.TestCase):
    """Contrato: load_data(path) -> (kms, prices), duas listas de float."""

    def setUp(self):
        from data import load_data
        self.load_data = load_data

    def test_retorna_duas_listas(self):
        kms, prices = self.load_data(DATA)
        self.assertIsInstance(kms, list)
        self.assertIsInstance(prices, list)

    def test_pula_cabecalho_e_le_24_linhas(self):
        kms, prices = self.load_data(DATA)
        self.assertEqual(len(kms), 24)
        self.assertEqual(len(prices), 24)

    def test_primeira_e_ultima_linha_na_ordem(self):
        kms, prices = self.load_data(DATA)
        self.assertEqual((kms[0], prices[0]), (240000.0, 3650.0))
        self.assertEqual((kms[-1], prices[-1]), (61789.0, 8290.0))

    def test_tudo_e_float(self):
        kms, prices = self.load_data(DATA)
        for valor in kms + prices:
            self.assertIs(type(valor), float)

    def test_arquivo_inexistente_levanta_erro(self):
        # load_data não imprime nem sai: quem decide o que fazer é quem chamou.
        with self.assertRaises(FileNotFoundError):
            self.load_data("nao_existe.csv")

    def test_csv_pequeno(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv",
                                         delete=False) as f:
            f.write("km,price\n10,20\n30,40\n")
        try:
            self.assertEqual(self.load_data(f.name),
                             ([10.0, 30.0], [20.0, 40.0]))
        finally:
            os.remove(f.name)


class TestEstimatePrice(unittest.TestCase):
    """Contrato: estimate_price(mileage, theta0, theta1) -> θ0 + θ1 * mileage."""

    def setUp(self):
        from predict import estimate_price
        self.estimate_price = estimate_price

    def test_thetas_zerados_preveem_zero(self):
        # O subject: antes do treino, theta0 = theta1 = 0.
        self.assertEqual(self.estimate_price(240000, 0.0, 0.0), 0.0)

    def test_so_intercepto(self):
        self.assertEqual(self.estimate_price(123456, 8000.0, 0.0), 8000.0)

    def test_reta_que_desce(self):
        # 8000 + (-0.02 * 100000) = 6000
        self.assertAlmostEqual(self.estimate_price(100000, 8000.0, -0.02),
                               6000.0)

    def test_km_zero_devolve_theta0(self):
        self.assertEqual(self.estimate_price(0, 8500.0, -0.02), 8500.0)


class TestLoadThetas(unittest.TestCase):
    """Contrato: load_thetas(path) -> (theta0, theta1); (0.0, 0.0) se não há
    arquivo. O formato do arquivo fica pro dia do train.py."""

    def test_sem_arquivo_devolve_zeros(self):
        from predict import load_thetas
        self.assertEqual(load_thetas("thetas_que_nao_existem.txt"),
                         (0.0, 0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
