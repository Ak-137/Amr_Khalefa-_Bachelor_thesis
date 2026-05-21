"""
Conditional AE: same train / test / Selective Mahalanobis pipeline as DCASE2023T2AE.
Domain labels from filenames (source vs target); condition injected in ConditionalAENet.
"""

from networks.dcase2023t2_ae.dcase2023t2_ae import DCASE2023T2AE
from networks.criterion.mahala import loss_function_mahala
from networks.conditional_ae.conditional_ae_net import ConditionalAENet
from networks.conditional_ae.domain_loader import DomainLoaderWrapper


class DCASE2023T2CondAE(DCASE2023T2AE):
    def __init__(self, args, train, test):
        super().__init__(args=args, train=train, test=test)
        self.train_loader = DomainLoaderWrapper(self.train_loader, self.model)
        self.valid_loader = DomainLoaderWrapper(self.valid_loader, self.model)

    def init_model(self):
        self.block_size = self.data.height
        return ConditionalAENet(input_dim=self.data.input_dim, block_size=self.block_size)

    def eval(
        self,
        test_loader,
        y_pred,
        anomaly_score_list,
        decision_result_list,
        domain_list,
        y_true,
        decision_threshold,
        mode,
        inv_cov_source,
        inv_cov_target,
    ):
        for j, batch in enumerate(test_loader):
            data = batch[0]
            data = data.to(self.device).float()
            y_true.append(batch[1][0].item())
            basename = batch[3][0]
            self.model.set_domain_from_names([basename], self.device, data.dtype)

            recon_data, _ = self.model(data)

            if self.args.score == "MAHALA":
                loss_source, num = loss_function_mahala(
                    recon_x=recon_data,
                    x=data,
                    block_size=self.block_size,
                    cov=inv_cov_source,
                    use_precision=True,
                    reduction=False,
                )
                loss_source = self.loss_reduction(
                    score=self.loss_reduction_1d(loss_source), n_loss=num
                )

                loss_target, num = loss_function_mahala(
                    recon_x=recon_data,
                    x=data,
                    block_size=self.block_size,
                    cov=inv_cov_target,
                    use_precision=True,
                    reduction=False,
                )
                loss_target = self.loss_reduction(
                    score=self.loss_reduction_1d(loss_target), n_loss=num
                )
                y_pred.append(min(loss_target.item(), loss_source.item()))
            else:
                y_pred.append(self.loss_fn(recon_x=recon_data, x=data).mean().item())

            anomaly_score_list.append([basename, y_pred[-1]])
            if y_pred[-1] > decision_threshold:
                decision_result_list.append([basename, 1])
            else:
                decision_result_list.append([basename, 0])

            if mode:
                domain_list.append("target" if "target" in basename else "source")
        return y_pred, anomaly_score_list, decision_result_list, domain_list
