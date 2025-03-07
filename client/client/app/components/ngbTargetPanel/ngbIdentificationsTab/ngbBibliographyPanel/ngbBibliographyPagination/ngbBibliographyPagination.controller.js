const PAGE_DEEPNESS = 3;

export default class ngbBibliographyPaginationController {


    static get UID() {
        return 'ngbBibliographyPaginationController';
    }

    constructor($scope, $timeout, dispatcher, ngbBibliographyPanelService) {
        Object.assign(this, {$scope, $timeout, ngbBibliographyPanelService});
        this.pages = this.getPages();

        const refresh = this.refresh.bind(this);
        dispatcher.on('target:identification:publications:results:updated', refresh);
        $scope.$on('$destroy', () => {
            dispatcher.removeListener('target:identification:publications:results:updated', refresh);
        });
    }

    get loadingPublications() {
        return this.ngbBibliographyPanelService.loadingPublications;
    }

    get totalPublications() {
        return this.ngbBibliographyPanelService.totalPublications;
    }

    refresh() {
        this.pages = this.getPages();
        this.$timeout(() => this.$scope.$apply());
    }

    async setPage(page) {
        if (this.totalPages === undefined || page < 1 || page > this.totalPages) {
            return;
        }
        const success = await this.ngbBibliographyPanelService.getDataOnPage(page);
        if (success) {
            this.pages = this.getPages();
        }
    }

    get totalPages() {
        return this.ngbBibliographyPanelService.totalPages;
    }

    get currentPage() {
        return this.ngbBibliographyPanelService.currentPage;
    }

    getPages() {
        const totalPages = this.totalPages;
        const currentPage = this.currentPage;
        if (totalPages === undefined || currentPage === undefined) {
            return [];
        }

        let minimumPage = Math.max(1, currentPage - PAGE_DEEPNESS);
        let maximumPage = Math.min(totalPages, currentPage + PAGE_DEEPNESS);
        minimumPage = Math.max(1, Math.min(minimumPage, maximumPage - (PAGE_DEEPNESS * 2)));
        maximumPage = Math.min(Math.max(maximumPage, minimumPage + (PAGE_DEEPNESS * 2)), totalPages);

        const pages = [];
        for (let i = minimumPage; i <= maximumPage; i++) {
            if (i === minimumPage && minimumPage > 1) {
                pages.push({
                    isFirst: true,
                    isLast: false,
                    value: 1
                });
            } else if (i === maximumPage && maximumPage < totalPages) {
                pages.push({
                    isFirst: false,
                    isLast: true,
                    value: totalPages
                });
            } else {
                pages.push({
                    isFirst: false,
                    isLast: false,
                    value: i
                });
            }
        }
        return pages;
    }
}
