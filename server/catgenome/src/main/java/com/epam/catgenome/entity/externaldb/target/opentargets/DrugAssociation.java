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
package com.epam.catgenome.entity.externaldb.target.opentargets;

import com.epam.catgenome.entity.externaldb.target.Association;
import com.epam.catgenome.entity.externaldb.target.UrlEntity;
import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class DrugAssociation extends Association {
    public static final String URL_PATTERN = "https://platform.opentargets.org/drug/%s";
    private String geneSymbol;
    private String geneName;
    private UrlEntity disease;
    private String drugType;
    private String mechanismOfAction;
    private String actionType;
    private String phase;
    private String status;
    private UrlEntity source;
    @Builder
    public DrugAssociation(String id, String name, String url, String geneId, String target,
                           String geneSymbol, String geneName, UrlEntity disease, String drugType,
                           String mechanismOfAction, String actionType, String phase, String status, UrlEntity source) {
        super(id, name, url, geneId, target);
        this.geneSymbol = geneSymbol;
        this.geneName = geneName;
        this.disease = disease;
        this.drugType = drugType;
        this.mechanismOfAction = mechanismOfAction;
        this.actionType = actionType;
        this.phase = phase;
        this.status = status;
        this.source = source;
    }
}
