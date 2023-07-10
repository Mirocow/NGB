/*
 * MIT License
 *
 * Copyright (c) 2016-2023 EPAM Systems
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

package com.epam.catgenome.manager.externaldb;

import com.epam.catgenome.controller.vo.externaldb.NCBISummaryVO;
import com.epam.catgenome.manager.externaldb.ncbi.NCBIDataManager;
import com.epam.catgenome.manager.externaldb.ncbi.NCBIGeneIdsManager;
import com.epam.catgenome.manager.externaldb.ncbi.NCBIGeneManager;
import com.epam.catgenome.manager.externaldb.ncbi.util.NCBIDatabase;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;


@Service
@RequiredArgsConstructor
@Slf4j
public class PudMedService {

    private final NCBIGeneIdsManager ncbiGeneIdsManager;
    private final NCBIGeneManager ncbiGeneManager;
    private final NCBIDataManager ncbiDataManager;

    @SneakyThrows
    public SearchResult<NCBISummaryVO> fetchPubMedArticles(final List<String> geneIds) {
        final Collection<String> entrezIds = ncbiGeneIdsManager.searchByEnsemblIds(geneIds).keySet();
        final List<NCBISummaryVO> articles = entrezIds.stream()
                .map(ncbiGeneManager::fetchPubmedData)
                .flatMap(List::stream)
                .collect(Collectors.toList());
        final SearchResult<NCBISummaryVO> result = new SearchResult<>();
        result.setItems(articles);
        result.setTotalCount(articles.size());
        return result;
    }

    @SneakyThrows
    public String getArticleAbstracts(final List<String> pmcIds) {
        return ncbiDataManager.fetchTextById(NCBIDatabase.PUBMED.name(),
                String.join(",", pmcIds), "Abstract");
    }
}