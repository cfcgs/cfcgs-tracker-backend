from sqlalchemy.orm import Session
from src.cfcgs_tracker.models import Region, Country


def create_initial_regions(db: Session):
    """
    Verifica se as regiões iniciais existem no banco e as cria se necessário.
    Esta função é 'idempotente', ou seja, pode ser executada várias vezes
    sem criar duplicatas.
    """
    GEOGRAPHIC_DATA = {
        "África Subsaariana": [
            "Costa do Marfim",
            "Djibuti",
            "Guiné Equatorial",
            "Eritreia",
            "Essuatíni",
            "Etiópia",
            "Gabão",
            "Gâmbia",
            "Gana",
            "Guiné",
            "Guiné-Bissau",
            "Quênia",
            "Lesoto",
            "Libéria",
            "Madagáscar",
            "Malawi",
            "Mali",
            "Mauritânia",
            "Maurício",
            "Moçambique",
            "Namíbia",
            "Níger",
            "Nigéria",
            "Ruanda",
            "São Tomé e Príncipe",
            "Senegal",
            "Seicheles",
            "Serra Leoa",
            "Somália",
            "África do Sul",
            "Sudão do Sul",
            "Sudão",
            "Tanzânia",
            "Togo",
            "Uganda",
            "Zâmbia",
            "Zimbábue",
        ],
        "Oriente Médio e Norte da África": [
            "Argélia",
            "Bahrein",
            "Egito",
            "Irã",
            "Iraque",
            "Israel",
            "Jordânia",
            "Kuwait",
            "Líbano",
            "Líbia",
            "Marrocos",
            "Omã",
            "Catar",
            "Arábia Saudita",
            "Síria",
            "Tunísia",
            "Emirados Árabes Unidos",
            "Cisjordânia e Faixa de Gaza",
            "Iémen",
        ],
        "Leste Asiático e Pacífico": [
            "Brunei",
            "Camboja",
            "China",
            "Fiji",
            "Indonésia",
            "Japão",
            "Kiribati",
            "Laos",
            "Malásia",
            "Ilhas Marshall",
            "Micronésia",
            "Mongólia",
            "Mianmar",
            "Nauru",
            "Nova Zelândia",
            "Palau",
            "Papua-Nova Guiné",
            "Filipinas",
            "Coreia do Sul",
            "Samoa",
            "Cingapura",
            "Ilhas Salomão",
            "Tailândia",
            "Timor-Leste",
            "Tonga",
            "Tuvalu",
            "Vanuatu",
            "Vietnã",
        ],
        "Europa e Ásia Central": [
            "Albânia",
            "Armênia",
            "Azerbaijão",
            "Belarus",
            "Bósnia e Herzegovina",
            "Bulgária",
            "Croácia",
            "Geórgia",
            "Cazaquistão",
            "Kosovo",
            "Quirguistão",
            "Moldávia",
            "Montenegro",
            "Macedônia do Norte",
            "Romênia",
            "Rússia",
            "Sérvia",
            "Tajiquistão",
            "Turcomenistão",
            "Turquia",
            "Ucrânia",
            "Uzbequistão",
        ],
        "América Latina e Caribe": [
            "Antígua e Barbuda",
            "Argentina",
            "Bahamas",
            "Barbados",
            "Belize",
            "Bolívia",
            "Brasil",
            "Chile",
            "Colômbia",
            "Costa Rica",
            "Cuba",
            "Dominica",
            "República Dominicana",
            "Equador",
            "El Salvador",
            "Granada",
            "Guatemala",
            "Guiana",
            "Haiti",
            "Honduras",
            "Jamaica",
            "México",
            "Nicarágua",
            "Panamá",
            "Paraguai",
            "Peru",
            "São Cristóvão e Nevis",
            "Santa Lúcia",
            "São Vicente e Granadinas",
            "Suriname",
            "Trinidad e Tobago",
            "Uruguai",
            "Venezuela",
        ],
        "Sul da Ásia": [
            "Afeganistão",
            "Bangladesh",
            "Butão",
            "Índia",
            "Maldivas",
            "Nepal",
            "Paquistão",
            "Sri Lanka",
        ],
    }

    print(
        "Verificando e criando dados geográficos iniciais (Regiões e Países)..."
    )

    # 2. Primeiro, cria as Regiões que não existem e as guarda em um mapa
    regions_map = {}
    for region_name in GEOGRAPHIC_DATA.keys():
        region_obj = db.query(Region).filter_by(name=region_name).first()
        if not region_obj:
            region_obj = Region(name=region_name)
            db.add(region_obj)
        regions_map[region_name] = region_obj

    # db.flush() envia os novos registros para o DB e atribui IDs, dentro da mesma transação
    db.flush()

    # 3. Em seguida, cria os Países e os associa às regiões corretas
    for region_name, countries_list in GEOGRAPHIC_DATA.items():
        region_obj = regions_map[region_name]
        for country_name in countries_list:
            country_exists = (
                db.query(Country).filter_by(name=country_name).first()
            )
            if not country_exists:
                new_country = Country(name=country_name)
                new_country.region = region_obj
                db.add(new_country)

    db.commit()
    print("Seeding de dados geográficos concluído.")
