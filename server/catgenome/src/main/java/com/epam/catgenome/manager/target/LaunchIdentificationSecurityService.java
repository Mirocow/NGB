/*
 * MIT License
 *
 * Copyright (c) 2023 EPAM Systems
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
package com.epam.catgenome.manager.target;

import com.epam.catgenome.controller.vo.externaldb.NCBISummaryVO;
import com.epam.catgenome.controller.vo.target.PublicationSearchRequest;
import com.epam.catgenome.controller.vo.target.StructuresSearchRequest;
import com.epam.catgenome.entity.externaldb.ncbi.GeneInfo;
import com.epam.catgenome.entity.externaldb.target.opentargets.BareDisease;
import com.epam.catgenome.entity.externaldb.target.pharmgkb.PharmGKBDisease;
import com.epam.catgenome.entity.externaldb.target.ttd.TTDDiseaseAssociation;
import com.epam.catgenome.entity.externaldb.target.ttd.TTDDrugAssociation;
import com.epam.catgenome.entity.target.GeneRefSection;
import com.epam.catgenome.manager.externaldb.bindings.rcsbpbd.dto.Structure;
import com.epam.catgenome.manager.externaldb.target.AssociationSearchRequest;
import com.epam.catgenome.manager.externaldb.target.opentargets.DrugFieldValues;
import com.epam.catgenome.manager.externaldb.target.dgidb.DGIDBDrugFieldValues;
import com.epam.catgenome.entity.externaldb.target.dgidb.DGIDBDrugAssociation;
import com.epam.catgenome.entity.externaldb.target.opentargets.DiseaseAssociationAggregated;
import com.epam.catgenome.entity.externaldb.target.opentargets.DrugAssociation;
import com.epam.catgenome.entity.externaldb.target.pharmgkb.PharmGKBDrug;
import com.epam.catgenome.entity.target.IdentificationRequest;
import com.epam.catgenome.entity.target.TargetIdentificationResult;
import com.epam.catgenome.exception.ExternalDbUnavailableException;
import com.epam.catgenome.manager.externaldb.SearchResult;
import com.epam.catgenome.manager.externaldb.target.pharmgkb.PharmGKBDrugFieldValues;
import com.epam.catgenome.manager.externaldb.target.ttd.TTDDiseaseFieldValues;
import com.epam.catgenome.manager.externaldb.target.ttd.TTDDrugFieldValues;
import lombok.RequiredArgsConstructor;
import org.apache.lucene.queryparser.classic.ParseException;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.List;

import static com.epam.catgenome.security.acl.SecurityExpressions.ROLE_ADMIN;
import static com.epam.catgenome.security.acl.SecurityExpressions.ROLE_USER;

@Service
@RequiredArgsConstructor
public class LaunchIdentificationSecurityService {

    private final LaunchIdentificationManager manager;

    @PreAuthorize(ROLE_USER)
    public TargetIdentificationResult launchIdentification(final IdentificationRequest request)
            throws ExternalDbUnavailableException, ParseException, IOException {
        return manager.launchIdentification(request);
    }

    @PreAuthorize(ROLE_USER)
    public SearchResult<DGIDBDrugAssociation> getDGIDbDrugs(final AssociationSearchRequest request)
            throws ParseException, IOException {
        return manager.getDGIDbDrugs(request);
    }

    @PreAuthorize(ROLE_USER)
    public SearchResult<PharmGKBDrug> getPharmGKBDrugs(final AssociationSearchRequest request)
            throws ParseException, IOException {
        return manager.getPharmGKBDrugs(request);
    }

    @PreAuthorize(ROLE_USER)
    public SearchResult<PharmGKBDisease> getPharmGKBDiseases(final AssociationSearchRequest request)
            throws ParseException, IOException {
        return manager.getPharmGKBDiseases(request);
    }

    @PreAuthorize(ROLE_USER)
    public SearchResult<DrugAssociation> getOpenTargetsDrugs(final AssociationSearchRequest request)
            throws IOException, ParseException {
        return manager.getOpenTargetsDrugs(request);
    }

    @PreAuthorize(ROLE_USER)
    public SearchResult<DiseaseAssociationAggregated> getOpenTargetsDiseases(final AssociationSearchRequest request)
            throws IOException, ParseException {
        return manager.getOpenTargetsDiseases(request);
    }

    @PreAuthorize(ROLE_USER)
    public List<DiseaseAssociationAggregated> getAllOpenTargetsDiseases(final AssociationSearchRequest request)
            throws IOException, ParseException {
        return manager.getAllOpenTargetsDiseases(request);
    }

    @PreAuthorize(ROLE_ADMIN)
    public void importOpenTargetsData(final String path) throws IOException, ParseException {
        manager.importOpenTargetsData(path);
    }

    @PreAuthorize(ROLE_ADMIN)
    public void importPharmGKBData(final String genePath, final String drugPath,
                                   final String drugAssociationPath, final String diseaseAssociationPath)
            throws IOException, ParseException {
        manager.importPharmGKBData(genePath, drugPath, drugAssociationPath, diseaseAssociationPath);
    }

    @PreAuthorize(ROLE_ADMIN)
    public void importDGIdbData(final String path) throws IOException, ParseException {
        manager.importDGIdbData(path);
    }

    @PreAuthorize(ROLE_ADMIN)
    public void importTTDData(final String drugsPath, final String targetsPath, final String diseasesPath)
            throws IOException, ParseException {
        manager.importTTDData(drugsPath, targetsPath, diseasesPath);
    }

    @PreAuthorize(ROLE_USER)
    public PharmGKBDrugFieldValues getPharmGKBDrugFieldValues(final Long targetId, final List<String> geneIds)
            throws IOException, ParseException {
        return manager.getPharmGKBDrugFieldValues(targetId, geneIds);
    }

    @PreAuthorize(ROLE_USER)
    public DGIDBDrugFieldValues getDGIDBDrugFieldValues(final Long targetId, final List<String> geneIds)
            throws IOException, ParseException {
        return manager.getDGIDBDrugFieldValues(targetId, geneIds);
    }

    @PreAuthorize(ROLE_USER)
    public DrugFieldValues getDrugFieldValues(final Long targetId, final List<String> geneIds)
            throws IOException, ParseException {
        return manager.getDrugFieldValues(targetId, geneIds);
    }

    @PreAuthorize(ROLE_USER)
    public List<BareDisease> getDiseasesTree() throws IOException {
        return manager.getDiseasesTree();
    }

    @PreAuthorize(ROLE_USER)
    public SearchResult<NCBISummaryVO> getPublications(final PublicationSearchRequest request)
            throws ParseException, IOException, ExternalDbUnavailableException {
        return manager.getPublications(request);
    }

    @PreAuthorize(ROLE_USER)
    public String getArticleAbstracts(final PublicationSearchRequest request) throws ParseException, IOException {
        return manager.getArticlesAbstracts(request);
    }

    @PreAuthorize(ROLE_USER)
    public List<GeneRefSection> getGeneSequencesTable(final Long targetId,
                                                      final List<String> geneIds,
                                                      final Boolean getComments,
                                                      final Boolean includeLocal,
                                                      final Boolean includeAdditionalGenes)
            throws ParseException, IOException, ExternalDbUnavailableException {
        return manager.getGeneSequencesTable(targetId, geneIds, getComments, includeLocal, includeAdditionalGenes);
    }

    @PreAuthorize(ROLE_USER)
    public SearchResult<Structure> getStructures(final StructuresSearchRequest request)
            throws ParseException, IOException {
        return manager.getStructures(request);
    }

    @PreAuthorize(ROLE_USER)
    public List<GeneInfo> getGenes(final String prefix) throws ParseException, IOException {
        return manager.getGenes(prefix);
    }

    @PreAuthorize(ROLE_USER)
    public List<String> getDrugs(final Long targetId, final List<String> geneIds) throws ParseException, IOException {
        return manager.getDrugs(targetId, geneIds);
    }

    @PreAuthorize(ROLE_USER)
    public SearchResult<TTDDrugAssociation> getTTDDrugs(final AssociationSearchRequest request)
            throws ParseException, IOException {
        return manager.getTTDDrugs(request);
    }

    @PreAuthorize(ROLE_USER)
    public TTDDrugFieldValues getTTDDrugFieldValues(final Long targetId, final List<String> geneIds)
            throws ParseException, IOException {
        return manager.getTTDDrugFieldValues(targetId, geneIds);
    }

    @PreAuthorize(ROLE_USER)
    public SearchResult<TTDDiseaseAssociation> getTTDDiseases(final AssociationSearchRequest request)
            throws ParseException, IOException {
        return manager.getTTDDiseases(request);
    }

    @PreAuthorize(ROLE_USER)
    public TTDDiseaseFieldValues getTTDDiseaseFieldValues(final Long targetId, final List<String> geneIds)
            throws ParseException, IOException {
        return manager.getTTDDiseaseFieldValues(targetId, geneIds);
    }
}
