# -*- coding: utf-8 -*-
from src.models.ontomap.ontology.anatomy import MouseHumanOMDataset
from src.models.ontomap.ontology.biodiv import (
    EnvoSweetOMDataset,
    FishZooplanktonOMDataset,
    MacroalgaeMacrozoobenthosOMDataset,
    TaxrefldBacteriaNcbitaxonBacteriaOMDataset,
    TaxrefldChromistaNcbitaxonChromistaOMDataset,
    TaxrefldFungiNcbitaxonFungiOMDataset,
    TaxrefldPlantaeNcbitaxonPlantaeOMDataset,
    TaxrefldProtozoaNcbitaxonProtozoaOMDataset,
)
from src.models.ontomap.ontology.bioml import (
    NCITDOIDDiseaseOMDataset,
    OMIMORDODiseaseOMDataset,
    SNOMEDFMABodyOMDataset,
    SNOMEDNCITNeoplasOMDataset,
    SNOMEDNCITPharmOMDataset,
)
from src.models.ontomap.ontology.commonkg import NellDbpediaOMDataset, YagoWikidataOMDataset
from src.models.ontomap.ontology.mse import (
    MaterialInformationEMMOOMDataset,
    MaterialInformationMatOntoMDataset,
)
from src.models.ontomap.ontology.phenotype import DoidOrdoOMDataset, HpMpOMDataset

ontology_matching = {
    "anatomy": [MouseHumanOMDataset],
    "biodiv": [
        EnvoSweetOMDataset,
        FishZooplanktonOMDataset,
        MacroalgaeMacrozoobenthosOMDataset,
        TaxrefldBacteriaNcbitaxonBacteriaOMDataset,
        TaxrefldChromistaNcbitaxonChromistaOMDataset,
        TaxrefldFungiNcbitaxonFungiOMDataset,
        TaxrefldPlantaeNcbitaxonPlantaeOMDataset,
        TaxrefldProtozoaNcbitaxonProtozoaOMDataset,
    ],
    "phenotype": [DoidOrdoOMDataset, HpMpOMDataset],
    "commonkg": [NellDbpediaOMDataset, YagoWikidataOMDataset],
    "bio-ml": [
        NCITDOIDDiseaseOMDataset,
        OMIMORDODiseaseOMDataset,
        SNOMEDFMABodyOMDataset,
        SNOMEDNCITNeoplasOMDataset,
        SNOMEDNCITPharmOMDataset,
    ],
    "mse": [
        MaterialInformationEMMOOMDataset,
        MaterialInformationMatOntoMDataset
    ],
}

__all__ = ["ontology_matching"]
