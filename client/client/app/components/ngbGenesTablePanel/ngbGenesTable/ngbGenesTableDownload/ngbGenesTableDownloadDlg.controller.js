export default class ngbGenesTableDownloadDlgController {

    isLoading = false;
    includeHeader = false;
    formatList = [
        {
            name: 'CSV',
            mimeType: 'text/csv'
        },
        {
            name: 'TSV',
            mimeType: 'text/tab-separated-values'
        }
    ];
    downloadFormat;

    constructor(ngbGenesTableService, $mdDialog, projectContext) {
        this.ngbGenesTableService = ngbGenesTableService;
        this.$mdDialog = $mdDialog;
        this.projectContext = projectContext;

        this.downloadFormat = this.formatList[0];
    }

    static get UID() {
        return 'ngbGenesTableDownloadDlgController';
    }

    download(event) {
        this.isLoading = true;
        this.ngbGenesTableService.downloadFile(
            this.projectContext.reference.id,
            this.downloadFormat.name,
            this.includeHeader
        ).then(data => {
            const linkElement = document.createElement('a');
            try {
                const blob = new Blob([data], {type: this.downloadFormat.mimeType});
                const url = window.URL.createObjectURL(blob);

                linkElement.setAttribute('href', url);
                linkElement.setAttribute('download',
                    `GENES-${this.projectContext.reference.id}.${this.downloadFormat.name.toLowerCase()}`);

                const clickEvent = new MouseEvent('click', {
                    'view': window,
                    'bubbles': true,
                    'cancelable': false
                });
                linkElement.dispatchEvent(clickEvent);
                this.close();
            } catch (ex) {
                // eslint-disable-next-line no-console
                console.error(ex);
            }
            this.isLoading = false;
        });
        event.stopImmediatePropagation();
        return false;
    }

    close() {
        this.$mdDialog.hide();
    }
}